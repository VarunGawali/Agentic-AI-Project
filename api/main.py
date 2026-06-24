"""
FastAPI backend for the Hedging Assistant dashboard.

Endpoints:
  POST /recommend   — run the full agent pipeline, return recommendation JSON
  POST /stress-test — run deterministic scenario paths, return per-scenario costs
  GET  /health      — liveness probe
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hedging_assistant.data.loader import load_price_history, make_exposure_book
from hedging_assistant.contracts import (
    RiskAppetite, RunConfig, ForwardCurve,
    StrategyParams, StrategyType,
)
from hedging_assistant.agent.orchestrator import HedgingAgent
from hedging_assistant.engines.cost_simulator import simulate_cost
from hedging_assistant.engines.forecaster import forecast as run_forecast

app = FastAPI(title="Hedging Assistant API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    forward_price: float = Field(80.0, gt=0, description="USD/bbl locked-in price")
    barrels_per_period: float = Field(100_000, gt=0)
    horizon: int = Field(6, ge=1, le=24)
    max_hedge: float = Field(1.0, ge=0, le=1)
    cvar_weight: float = Field(1.0, ge=0, le=5)
    model: str = Field("normal", pattern="^(normal|student-t|hmm|xgb-garch-t)$")
    n_paths: int = Field(2000, ge=100, le=20000)

class StressRequest(BaseModel):
    forward_price: float = Field(80.0, gt=0)
    barrels_per_period: float = Field(100_000, gt=0)
    horizon: int = Field(6, ge=1, le=24)
    hedge_fraction: float = Field(0.5, ge=0, le=1)
    custom_shock_pct: float = Field(40.0, description="Custom shock % (positive=spike)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_scenario_paths(base_price: float, horizon: int, shock_pct: float,
                           pattern: str = "linear") -> np.ndarray:
    """Return shape (1, horizon) deterministic path for stress testing."""
    path = np.full(horizon, base_price)
    shock = base_price * shock_pct / 100
    if pattern == "linear":
        path = base_price + np.linspace(0, shock, horizon)
    elif pattern == "spike_recover":
        mid = horizon // 2
        path[:mid] = base_price + np.linspace(0, shock, mid)
        path[mid:] = path[mid-1] + np.linspace(0, -shock * 0.6, horizon - mid)
    elif pattern == "crash":
        path = base_price + np.linspace(0, shock, horizon)  # shock is negative
    elif pattern == "plateau":
        path[:2] = base_price
        path[2:] = base_price + shock
    return path.reshape(1, -1)


SCENARIOS = {
    "2022 Spike":       {"shock_pct": +80,  "pattern": "linear"},
    "2008 Crash":       {"shock_pct": -60,  "pattern": "crash"},
    "COVID Collapse":   {"shock_pct": -75,  "pattern": "spike_recover"},
    "Geopolitical":     {"shock_pct": +40,  "pattern": "plateau"},
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    history = load_price_history(symbol="WTI")

    risk = RiskAppetite(
        max_hedge=req.max_hedge,
        w_cvar=req.cvar_weight,
        cvar_alpha=0.95,
    )
    run_config = RunConfig(
        n_paths=req.n_paths,
        seed=42,
        frequency="M",
        distribution="student-t" if req.model == "student-t" else "normal",
        use_regime=(req.model == "hmm"),
        model="xgb-garch-t" if req.model == "xgb-garch-t" else "gbm",
    )

    agent = HedgingAgent(risk=risk, run_config=run_config)
    exposure = make_exposure_book(
        barrels_per_period=req.barrels_per_period,
        horizon=req.horizon,
    )

    try:
        rec = agent.recommend(history, exposure, req.forward_price)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build forecast fan data (P10/P50/P90 per period)
    fc = agent.forecast(history, req.horizon)
    paths = np.asarray(fc.paths)
    fan = {
        "p10": np.percentile(paths, 10, axis=0).tolist(),
        "p25": np.percentile(paths, 25, axis=0).tolist(),
        "p50": np.percentile(paths, 50, axis=0).tolist(),
        "p75": np.percentile(paths, 75, axis=0).tolist(),
        "p90": np.percentile(paths, 90, axis=0).tolist(),
    }

    # Historical prices (last 24 periods for chart context)
    hist_tail = history.prices[-24:].tolist()
    hist_dates = [str(d)[:10] for d in history.dates[-24:]]

    # Cost distribution (histogram buckets)
    costs = rec.cost.costs
    hist_counts, bin_edges = np.histogram(costs / 1e6, bins=40)
    no_hedge_params = StrategyParams(StrategyType.STAGGERED, base_fraction=0.0)
    no_hedge_cost = simulate_cost(fc, exposure, no_hedge_params, req.forward_price)
    nh_counts, nh_edges = np.histogram(no_hedge_cost.costs / 1e6, bins=40)

    # Candidate table
    candidates = []
    for r in rec.trace:
        candidates.append({
            "hedge_pct": round(r.params.base_fraction * 100, 1),
            "expected_cost": round(r.score.cost / 1e6, 2),
            "cvar": round(r.score.cvar / 1e6, 2),
            "blended": round(r.score.blended / 1e6, 2),
            "accepted": r.accepted,
        })

    return {
        "rationale": rec.rationale,
        "hedge_fraction": round(rec.policy.params.base_fraction * 100, 1),
        "model_name": fc.model_name,
        "regime": "N/A",  # populated if HMM used
        "cost": {
            "mean": round(rec.cost.mean / 1e6, 2),
            "p10": round(rec.cost.p10 / 1e6, 2),
            "p50": round(rec.cost.p50 / 1e6, 2),
            "p90": round(rec.cost.p90 / 1e6, 2),
            "cvar": round(rec.cost.cvar / 1e6, 2),
            "ci_mean": [round(rec.cost.ci_mean[0]/1e6, 2), round(rec.cost.ci_mean[1]/1e6, 2)],
            "ci_cvar": [round(rec.cost.ci_cvar[0]/1e6, 2), round(rec.cost.ci_cvar[1]/1e6, 2)],
        },
        "policy_schedule": [round(f, 3) for f in rec.policy.hedge_fractions.tolist()],
        "forward_price": req.forward_price,
        "fan": fan,
        "history": {"dates": hist_dates, "prices": hist_tail},
        "cost_histogram": {
            "strategy": {"counts": hist_counts.tolist(), "edges": bin_edges.tolist()},
            "no_hedge": {"counts": nh_counts.tolist(), "edges": nh_edges.tolist()},
            "cvar_line": round(rec.cost.cvar / 1e6, 2),
        },
        "candidates": candidates,
        "run_config": {
            "n_paths": req.n_paths,
            "model": fc.model_name,
            "frequency": run_config.frequency,
        },
    }


@app.post("/stress-test")
def stress_test(req: StressRequest):
    history = load_price_history(symbol="WTI")
    exposure = make_exposure_book(
        barrels_per_period=req.barrels_per_period,
        horizon=req.horizon,
    )

    # Build a minimal PriceForecast wrapper for each scenario
    from hedging_assistant.contracts import PriceForecast
    params_hedged = StrategyParams(StrategyType.STAGGERED, base_fraction=req.hedge_fraction)
    params_none   = StrategyParams(StrategyType.STAGGERED, base_fraction=0.0)

    results = []

    all_scenarios = {**SCENARIOS, "Custom Shock": {"shock_pct": req.custom_shock_pct, "pattern": "linear"}}
    base = history.prices[-1]

    for name, cfg in all_scenarios.items():
        path = _build_scenario_paths(base, req.horizon, cfg["shock_pct"], cfg["pattern"])
        fc_stub = PriceForecast(paths=path, model_name="deterministic", frequency="M")

        cost_hedged   = simulate_cost(fc_stub, exposure, params_hedged,   req.forward_price)
        cost_no_hedge = simulate_cost(fc_stub, exposure, params_none,     req.forward_price)

        results.append({
            "scenario": name,
            "no_hedge_cost": round(cost_no_hedge.mean / 1e6, 2),
            "hedged_cost":   round(cost_hedged.mean   / 1e6, 2),
            "savings":       round((cost_no_hedge.mean - cost_hedged.mean) / 1e6, 2),
            "shock_pct":     cfg["shock_pct"],
        })

    return {"scenarios": results, "hedge_fraction": req.hedge_fraction}
