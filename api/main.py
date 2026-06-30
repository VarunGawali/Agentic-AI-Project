"""
FastAPI backend for the Hedging Assistant dashboard.

Endpoints:
    GET  /health
    POST /recommend
    POST /stress-test

Current integration:
    Uses the LangGraph workflow:
        assess -> forecast -> explore -> arbitrate -> explain

Phase 3:
    - Supports XGB-GARCH-t as the default forecaster
    - Supports 4-factor scoring controls:
        expected cost
        CVaR
        opportunity cost
        execution risk
    - Returns richer candidate trace for dashboard/governance
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow imports from project root when running from api/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("hedging_api")

from data.loader import load_price_history as load_price_history_df

from contracts import (
    PriceHistory,
    PriceForecast,
    ExposureBook,
    RiskAppetite,
    StrategyParams,
    StrategyType,
)

from agent.langgraph_workflow import run_agent
from engines.cost_simulator import simulate_cost

# ---------------------------------------------------------------------------
# Price history cache — refreshed at most once per hour to avoid repeated
# file/network I/O on every slider change.
# ---------------------------------------------------------------------------
_PRICE_CACHE: dict = {"df": None, "ts": 0.0}
_PRICE_CACHE_TTL = int(os.environ.get("PRICE_CACHE_TTL_SECONDS", "3600"))


def _get_price_history() -> "pd.DataFrame":
    now = time.monotonic()
    if _PRICE_CACHE["df"] is not None and (now - _PRICE_CACHE["ts"]) < _PRICE_CACHE_TTL:
        return _PRICE_CACHE["df"]
    df = load_price_history_df(symbol="WTI")
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0].sort_values("date")
    _PRICE_CACHE["df"] = df
    _PRICE_CACHE["ts"] = now
    logger.info("Price history refreshed (%d rows).", len(df))
    return df


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hedging Assistant API",
    version="1.0",
    description="Backend API for crude procurement and hedging decision assistant.",
)

_cors_origins_raw = os.environ.get("CORS_ALLOWED_ORIGINS", "")
_cors_origins = (
    [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    if _cors_origins_raw
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    logger.info("Hedging Assistant API starting up")
    logger.info(
        "Config: EIA_API_KEY=%s azure_blob=%s azure_openai=%s model_dir=%s",
        "set" if os.environ.get("EIA_API_KEY") else "MISSING",
        "set" if os.environ.get("AZURE_STORAGE_CONNECTION_STRING") else "not set",
        "set" if os.environ.get("AZURE_OPENAI_API_KEY") else "not set",
        os.environ.get("MODEL_DIR", "models"),
    )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    forward_price: float = Field(
        95.0,
        gt=0,
        description="Forward/hedge price in USD per barrel.",
    )
    barrels_per_period: float = Field(
        100_000,
        gt=0,
        description="Procurement volume per period.",
    )
    horizon: int = Field(
        6,
        ge=1,
        le=24,
        description="Forecast and procurement horizon.",
    )
    max_hedge: float = Field(
        1.0,
        ge=0,
        le=1,
        description="Maximum allowed hedge fraction.",
    )
    cvar_weight: float = Field(
        1.0,
        ge=0,
        le=5,
        description="Weight on CVaR downside risk.",
    )
    opportunity_weight: float = Field(
        0.5,
        ge=0,
        le=5,
        description="Weight on opportunity cost.",
    )
    execution_weight: float = Field(
        0.25,
        ge=0,
        le=5,
        description="Weight on execution risk.",
    )
    model: str = Field(
        "xgb-garch-t",
        pattern="^(xgb-garch-t|gbm|normal|student-t|hmm)$",
        description="Forecast model.",
    )
    n_paths: int = Field(
        4096,
        ge=256,
        le=20000,
        description="Number of Monte Carlo paths.",
    )
    frequency: str = Field(
        "M",
        pattern="^(D|W|M)$",
        description="Planning frequency: D, W, or M.",
    )
    calibration_window: int | None = Field(
        1000,
        ge=60,
        le=5000,
        description="Rows/observations used for calibration.",
    )


class StressRequest(BaseModel):
    forward_price: float = Field(95.0, gt=0)
    barrels_per_period: float = Field(100_000, gt=0)
    horizon: int = Field(6, ge=1, le=24)
    hedge_fraction: float = Field(0.5, ge=0, le=1)
    custom_shock_pct: float = Field(
        40.0,
        description="Custom shock percentage. Positive means price spike.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_exposure_book(
    barrels_per_period: float,
    horizon: int,
    period_label: str = "month",
) -> ExposureBook:
    """
    Build simple flat exposure book.
    """

    return ExposureBook(
        volumes=np.full(horizon, barrels_per_period, dtype=float),
        period_label=period_label,
    )


def _build_fan_data(paths: np.ndarray) -> dict:
    """
    Build P10/P25/P50/P75/P90 forecast fan data.
    """

    return {
        "p10": np.percentile(paths, 10, axis=0).tolist(),
        "p25": np.percentile(paths, 25, axis=0).tolist(),
        "p50": np.percentile(paths, 50, axis=0).tolist(),
        "p75": np.percentile(paths, 75, axis=0).tolist(),
        "p90": np.percentile(paths, 90, axis=0).tolist(),
    }


def _build_period_costs(
    forecast_obj,
    hedge_fractions: np.ndarray,
    volumes: np.ndarray,
    forward_price: float,
) -> dict:
    """
    Build median per-period hedged and no-hedge costs in millions.

    Returns:
        dict with hedged_cost and no_hedge_cost lists, one value per period.
    """

    paths = np.asarray(forecast_obj.paths, dtype=float)
    n_paths, horizon = paths.shape

    fwd = np.full(horizon, float(forward_price))
    fracs = np.asarray(hedge_fractions, dtype=float)

    hedged_per_period = (fracs * volumes * fwd) + ((1 - fracs) * volumes * np.median(paths, axis=0))
    no_hedge_per_period = volumes * np.median(paths, axis=0)

    return {
        "hedged_cost":   [round(float(v) / 1e6, 3) for v in hedged_per_period],
        "no_hedge_cost": [round(float(v) / 1e6, 3) for v in no_hedge_per_period],
    }


def _build_histogram(values: np.ndarray, bins: int = 40) -> dict:
    """
    Build histogram for dashboard plotting.
    Values are expected in raw dollars.
    Returned values are in millions.
    """

    counts, edges = np.histogram(values / 1e6, bins=bins)

    return {
        "counts": counts.tolist(),
        "edges": edges.tolist(),
    }


def _safe_ci(value: Any) -> list[float] | None:
    """
    Convert optional confidence interval tuple to JSON-safe list in millions.
    """

    if value is None:
        return None

    return [
        round(float(value[0]) / 1e6, 2),
        round(float(value[1]) / 1e6, 2),
    ]


def _json_safe(value: Any) -> Any:
    """
    Convert common non-JSON-safe values into JSON-safe equivalents.

    This prevents FastAPI serialization issues when assumptions contain
    numpy arrays, dataclasses, paths, or large model objects.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        safe = {}

        for key, item in value.items():
            # Avoid shipping large objects like full forecast object to frontend.
            if key in {"forecast_obj", "paths", "raw_paths"}:
                continue

            safe[str(key)] = _json_safe(item)

        return safe

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    # Dataclass-like fallback
    if hasattr(value, "__dict__"):
        return str(value)

    return str(value)


def _selected_hedge_pct(params: StrategyParams) -> float:
    """
    Return representative hedge percentage for dashboard summary.
    """

    if params.strategy_type == StrategyType.CVAR_LP and params.fixed_fractions is not None:
        return float(np.mean(params.fixed_fractions) * 100.0)

    return float(params.base_fraction * 100.0)


def _build_scenario_path(
    base_price: float,
    horizon: int,
    shock_pct: float,
    pattern: str = "linear",
) -> np.ndarray:
    """
    Return deterministic scenario path with shape (1, horizon).
    """

    shock = base_price * shock_pct / 100.0

    if pattern == "linear":
        path = base_price + np.linspace(0, shock, horizon)

    elif pattern == "spike_recover":
        mid = max(1, horizon // 2)
        path = np.empty(horizon, dtype=float)
        path[:mid] = base_price + np.linspace(0, shock, mid)
        path[mid:] = path[mid - 1] + np.linspace(0, -shock * 0.6, horizon - mid)

    elif pattern == "crash":
        path = base_price + np.linspace(0, shock, horizon)

    elif pattern == "plateau":
        path = np.full(horizon, base_price, dtype=float)
        if horizon > 2:
            path[2:] = base_price + shock
        else:
            path[:] = base_price + shock

    else:
        raise ValueError(f"Unknown scenario pattern: {pattern}")

    # Avoid invalid non-positive prices in severe crash scenarios.
    path = np.maximum(path, 1.0)

    return path.reshape(1, -1)


SCENARIOS = {
    "2022 Spike": {
        "shock_pct": 80.0,
        "pattern": "linear",
    },
    "2008 Crash": {
        "shock_pct": -60.0,
        "pattern": "crash",
    },
    "COVID Collapse": {
        "shock_pct": -75.0,
        "pattern": "crash",
    },
    "Geopolitical Spike": {
        "shock_pct": 40.0,
        "pattern": "plateau",
    },
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """
    Liveness probe — always returns 200 while the process is alive.
    """
    return {"status": "ok", "service": "hedging-assistant-api"}


@app.get("/ready")
def ready() -> dict:
    """
    Readiness probe — checks that critical dependencies are reachable.

    Returns 200 when all checks pass; 503 when any check fails.
    """
    from fastapi.responses import JSONResponse

    checks: dict[str, str] = {}
    ok = True

    # Check 1: XGBoost model artifact resolvable
    try:
        from engines.xgb_garch_forecaster import (
            DEFAULT_MODEL_PATH,
            _blob_configured,
            _download_artifact_from_blob,
        )
        blob_name = os.environ.get("XGB_MODEL_BLOB_NAME", "xgb_drift_model.json")
        if _blob_configured():
            data = _download_artifact_from_blob(blob_name)
            checks["xgb_model"] = "blob_ok" if data is not None else "blob_missing"
            if data is None:
                ok = False
        elif DEFAULT_MODEL_PATH.exists():
            checks["xgb_model"] = "local_ok"
        else:
            checks["xgb_model"] = "missing"
            ok = False
    except Exception as exc:
        checks["xgb_model"] = f"error: {exc}"
        ok = False

    # Check 2: Azure OpenAI configured (warn only — deterministic fallback available)
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if azure_key and azure_endpoint:
        checks["azure_openai"] = "configured"
    else:
        checks["azure_openai"] = "not_configured (deterministic fallback active)"

    # Check 3: EIA API key present
    if os.environ.get("EIA_API_KEY"):
        checks["eia_api_key"] = "present"
    else:
        checks["eia_api_key"] = "missing"
        ok = False

    # Check 4: Azure Blob Storage (optional)
    if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
        checks["azure_blob"] = "configured"
    else:
        checks["azure_blob"] = "not_configured (local fallback active)"

    status_code = 200 if ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if ok else "not_ready", "checks": checks},
    )


@app.post("/recommend")
def recommend(req: RecommendRequest) -> dict:
    """
    Run LangGraph hedging assistant and return dashboard-ready JSON.
    """

    try:
        df = _get_price_history()

        history = PriceHistory(
            dates=df["date"].to_numpy(),
            prices=df["price"].to_numpy(dtype=float),
            symbol="WTI",
        )

        _period_label = {"D": "day", "W": "week", "M": "month"}.get(
            req.frequency.upper(), "month"
        )

        exposure = make_exposure_book(
            barrels_per_period=req.barrels_per_period,
            horizon=req.horizon,
            period_label=_period_label,
        )

        risk = RiskAppetite(
            w_cost=1.0,
            w_cvar=req.cvar_weight,
            w_opportunity=req.opportunity_weight,
            w_execution=req.execution_weight,
            cvar_alpha=0.95,
            max_hedge=req.max_hedge,
        )

        # Model mapping:
        # xgb-garch-t -> Phase 3 forecaster
        # student-t   -> legacy GBM with Student-t shocks
        # normal/gbm  -> legacy GBM normal shocks
        # hmm         -> legacy GBM with regime hook
        if req.model == "student-t":
            distribution = "student-t"
            model = "gbm"
            use_regime = False
        elif req.model == "hmm":
            distribution = "normal"
            model = "gbm"
            use_regime = True
        elif req.model in {"normal", "gbm"}:
            distribution = "normal"
            model = "gbm"
            use_regime = False
        else:
            distribution = "normal"
            model = "xgb-garch-t"
            use_regime = False

        # NOTE:
        # This assumes run_agent has been/will be updated to accept `model`.
        recommendation = run_agent(
            history=history,
            exposure=exposure,
            risk=risk,
            forward_price=req.forward_price,
            frequency=req.frequency,
            n_paths=req.n_paths,
            seed=42,
            calibration_window=req.calibration_window,
            distribution=distribution,
            use_regime=use_regime,
            model=model,
        )

        costs = np.asarray(recommendation.cost.costs, dtype=float)

        assumptions = getattr(recommendation, "assumptions", {}) or {}

        forecast_obj = assumptions.get("forecast_obj")
        fan = assumptions.get("fan")

        if fan is None and forecast_obj is not None:
            try:
                fan = _build_fan_data(np.asarray(forecast_obj.paths, dtype=float))
            except Exception:
                fan = None

        candidate_rows = []

        for record in recommendation.trace:
            candidate_rows.append(
                {
                    "strategy_type": record.params.strategy_type.value,
                    "hedge_pct": round(_selected_hedge_pct(record.params), 1),
                    "expected_cost": round(record.score.cost / 1e6, 2),
                    "cvar": round(record.score.cvar / 1e6, 2),
                    "opportunity_cost": round(record.score.opportunity_cost / 1e6, 2),
                    "execution_risk": round(record.score.execution_risk / 1e6, 2),
                    "blended": round(record.score.blended, 4),
                    "accepted": bool(record.accepted),
                    "note": record.note,
                }
            )

        hist_tail = history.prices[-24:].tolist()
        hist_dates = [str(d)[:10] for d in history.dates[-24:]]

        ci_mean = getattr(recommendation.cost, "ci_mean", None)
        ci_cvar = getattr(recommendation.cost, "ci_cvar", None)

        cost_histogram = assumptions.get("cost_histogram")

        if cost_histogram is None:
            cost_histogram = {
                "strategy": _build_histogram(costs),
                "cvar_line": round(recommendation.cost.cvar / 1e6, 2),
            }

        period_costs = None
        if forecast_obj is not None:
            try:
                period_costs = _build_period_costs(
                    forecast_obj=forecast_obj,
                    hedge_fractions=recommendation.policy.hedge_fractions,
                    volumes=exposure.volumes,
                    forward_price=req.forward_price,
                )
            except Exception:
                period_costs = None

        return {
            "rationale": recommendation.rationale,
            "hedge_fraction": round(
                _selected_hedge_pct(recommendation.policy.params),
                1,
            ),
            "selected_strategy_type": recommendation.policy.params.strategy_type.value,
            "policy_description": recommendation.policy.description,
            "policy_schedule": [
                round(float(f), 3)
                for f in recommendation.policy.hedge_fractions.tolist()
            ],
            "forward_price": req.forward_price,
            "model_name": assumptions.get("forecast_model", model),
            "frequency": assumptions.get("frequency", req.frequency),
            "n_paths": assumptions.get("n_paths", req.n_paths),
            "calibration_window": assumptions.get(
                "calibration_window",
                req.calibration_window,
            ),
            "fan": fan,
            "cost": {
                "mean": round(recommendation.cost.mean / 1e6, 2),
                "p10": round(recommendation.cost.p10 / 1e6, 2),
                "p50": round(recommendation.cost.p50 / 1e6, 2),
                "p90": round(recommendation.cost.p90 / 1e6, 2),
                "cvar": round(recommendation.cost.cvar / 1e6, 2),
                "ci_mean": _safe_ci(ci_mean),
                "ci_cvar": _safe_ci(ci_cvar),
            },
            "cost_histogram": cost_histogram,
            "period_costs": period_costs,
            "history": {
                "dates": hist_dates,
                "prices": hist_tail,
            },
            "candidates": candidate_rows,
            "assumptions": _json_safe(assumptions),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/stress-test")
def stress_test(req: StressRequest) -> dict:
    """
    Run deterministic scenario stress tests for selected hedge fraction.
    """

    try:
        df = _get_price_history()

        history = PriceHistory(
            dates=df["date"].to_numpy(),
            prices=df["price"].to_numpy(dtype=float),
            symbol="WTI",
        )

        exposure = make_exposure_book(
            barrels_per_period=req.barrels_per_period,
            horizon=req.horizon,
            period_label="month",
        )

        params_hedged = StrategyParams(
            strategy_type=StrategyType.STAGGERED,
            base_fraction=req.hedge_fraction,
            cap=1.0,
        )

        params_no_hedge = StrategyParams(
            strategy_type=StrategyType.STAGGERED,
            base_fraction=0.0,
            cap=1.0,
        )

        base_price = float(history.prices[-1])

        all_scenarios = {
            **SCENARIOS,
            "Custom Shock": {
                "shock_pct": req.custom_shock_pct,
                "pattern": "linear",
            },
        }

        results = []

        for name, cfg in all_scenarios.items():
            scenario_path = _build_scenario_path(
                base_price=base_price,
                horizon=req.horizon,
                shock_pct=cfg["shock_pct"],
                pattern=cfg["pattern"],
            )

            forecast_stub = PriceForecast(
                paths=scenario_path,
                model_name="deterministic-scenario",
                start_price=base_price,
                frequency="M",
            )

            cost_hedged = simulate_cost(
                forecast_obj=forecast_stub,
                exposure=exposure,
                params=params_hedged,
                forward_price=req.forward_price,
                cvar_alpha=0.95,
                mode="optimized",
            )

            cost_no_hedge = simulate_cost(
                forecast_obj=forecast_stub,
                exposure=exposure,
                params=params_no_hedge,
                forward_price=req.forward_price,
                cvar_alpha=0.95,
                mode="optimized",
            )

            results.append(
                {
                    "scenario": name,
                    "shock_pct": cfg["shock_pct"],
                    "path": scenario_path.flatten().round(2).tolist(),
                    "no_hedge_cost": round(cost_no_hedge.mean / 1e6, 2),
                    "hedged_cost": round(cost_hedged.mean / 1e6, 2),
                    "savings": round(
                        (cost_no_hedge.mean - cost_hedged.mean) / 1e6,
                        2,
                    ),
                }
            )

        return {
            "base_price": round(base_price, 2),
            "forward_price": req.forward_price,
            "hedge_fraction": req.hedge_fraction,
            "scenarios": results,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc