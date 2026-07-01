"""
Ensemble volatility forecaster (XGB-Ensemble-t).

Design (matches the multi-model sketch):
  Level anchor : driftless random walk (E[spot] = S0)
  Vol/Tail     : ensemble of {XGB-HAR vol, GJR-GARCH-t vol, EWMA/RiskMetrics vol}
  Scenario gen : driftless Student-t Monte Carlo, pooled as a mixture
  Selector     : per-member coverage-calibrated vol scales; weights combine members

Why a mixture (not a blended vol): pooling paths from each member propagates
*model uncertainty* into the distribution — when the members disagree the tails
widen, which is exactly what makes the CVaR the downstream optimizer minimizes more
robust to any single model being wrong.

Empirically the ensemble matches the single XGB-Vol-t on CRPS (the strongest member
dominates) with slightly better tail coverage; its value is robustness across
regimes and a well-calibrated distribution to feed the CVaR-LP.
"""

from __future__ import annotations

import logging

import numpy as np

from hedging_assistant.contracts import PriceHistory, PriceForecast
from hedging_assistant.engines.xgb_garch_forecaster import (
    prepare_daily_price_series,
    generate_student_t_shocks,
    fit_garch_t_on_residuals,
)
from hedging_assistant.engines.ml_vol_forecaster import (
    _feature_row,
    _train_vol_model,
    _predict_current_vol,
    _MIN_HISTORY,
    _DAILY_VOL_FLOOR,
    _DAILY_VOL_CEIL,
)

logger = logging.getLogger(__name__)

_ENSEMBLE_CACHE: dict = {}

# Per-member vol scale so each member's cone hits ~80% p10-p90 coverage
# (XGBoost regresses toward the mean and under-predicts vol; GARCH/EWMA less so).
_MEMBER_SCALE = {"xgb": 1.25, "garch": 1.15, "ewma": 1.20}
_MEMBER_WEIGHT = {"xgb": 0.5, "garch": 0.3, "ewma": 0.2}


def _ewma_vol(returns: np.ndarray, lam: float = 0.94) -> float:
    w = (1.0 - lam) * lam ** np.arange(len(returns))[::-1]
    return float(np.sqrt(np.sum(w * returns ** 2) / w.sum()))


def _garch_vol(returns: np.ndarray, horizon_days: int) -> float:
    """Analytic GARCH(1,1)-t horizon-average daily vol from the fitted params."""
    g = fit_garch_t_on_residuals(returns - returns.mean())
    lr_var = (g["long_run_vol"] * 100.0) ** 2
    s1 = g["last_sigma2_pct"]
    persistence = min(g["persistence"], 0.999)
    ks = np.arange(horizon_days)
    var_path = lr_var + persistence ** ks * (s1 - lr_var)
    return float(np.sqrt(max(var_path.mean(), 1e-12)) / 100.0)


def forecast_ensemble_t(
    history: PriceHistory,
    horizon: int,
    n_paths: int = 4096,
    seed: int | None = 42,
    calibration_window: int | None = 1500,
    daily_steps: int = 21,
    nu: float = 5.0,
) -> PriceForecast:
    """
    Driftless Student-t Monte Carlo whose volatility is a mixture of three
    forecasters. Paths are pooled across members (weighted by _MEMBER_WEIGHT) so
    the resulting distribution reflects model disagreement.
    """
    if horizon <= 0 or n_paths <= 0 or daily_steps <= 0:
        raise ValueError("horizon, n_paths, daily_steps must be positive")

    series = prepare_daily_price_series(history=history, calibration_window=None)
    prices = series.to_numpy(dtype=float)
    if calibration_window is not None and len(prices) > calibration_window:
        prices = prices[-calibration_window:]
    if len(prices) < _MIN_HISTORY:
        raise ValueError(f"XGB-Ensemble-t needs >= {_MIN_HISTORY} daily rows; got {len(prices)}.")

    returns = np.diff(np.log(prices))
    S0 = float(prices[-1])
    total_steps = horizon * daily_steps
    horizon_days = total_steps

    # --- member volatilities ---
    cache_key = (len(prices), round(S0, 4), horizon_days, seed)
    model = _ENSEMBLE_CACHE.get(cache_key)
    if model is None:
        model = _train_vol_model(returns, horizon_days, seed or 42)
        _ENSEMBLE_CACHE[cache_key] = model

    sig = {
        "xgb": _predict_current_vol(model, returns),
        "garch": float(np.clip(_garch_vol(returns[-750:], horizon_days), _DAILY_VOL_FLOOR, _DAILY_VOL_CEIL)),
        "ewma": float(np.clip(_ewma_vol(returns[-250:]), _DAILY_VOL_FLOOR, _DAILY_VOL_CEIL)),
    }

    # --- mixture: split paths across members by weight, simulate driftless ---
    members = ["xgb", "garch", "ewma"]
    counts = {m: int(round(_MEMBER_WEIGHT[m] * n_paths)) for m in members}
    counts["xgb"] += n_paths - sum(counts.values())  # fix rounding to exactly n_paths

    period_idx = (np.arange(1, horizon + 1) * daily_steps) - 1
    chunks = []
    for m in members:
        nm = counts[m]
        if nm <= 0:
            continue
        shocks = generate_student_t_shocks(
            n_paths=nm, total_steps=total_steps, nu=nu, seed=seed, use_sobol=True
        )
        log_returns = (_MEMBER_SCALE[m] * sig[m]) * shocks
        daily_prices = np.exp(np.log(S0) + np.cumsum(log_returns, axis=1))
        chunks.append(daily_prices[:, period_idx])

    paths = np.vstack(chunks)

    logger.info(
        "[ensemble_forecaster] sig xgb=%.4f garch=%.4f ewma=%.4f -> "
        "E[term]/S0=%.3f median[term]/S0=%.3f",
        sig["xgb"], sig["garch"], sig["ewma"],
        float(paths[:, -1].mean() / S0), float(np.median(paths[:, -1]) / S0),
    )

    return PriceForecast(
        paths=paths,
        model_name="XGB-Ensemble-t",
        start_price=S0,
        mu=0.0,
        sigma=float(sum(_MEMBER_WEIGHT[m] * _MEMBER_SCALE[m] * sig[m] for m in members)),
        frequency="M",
        seed=seed,
        calibration_window=len(prices),
    )
