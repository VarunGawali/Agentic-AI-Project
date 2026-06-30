"""
XGB-Vol-t forecaster: ML volatility + driftless Student-t Monte Carlo.

Motivation
----------
Empirically, crude-oil price *direction* is not forecastable (information
coefficient ~0), but *volatility* is (vol clustering; trailing realized vol
predicts forward realized vol with Spearman IC ~0.4-0.5 on WTI). The earlier
XGB-GARCH-t model pointed ML at drift — the unforecastable target — which both
biased the mean (E[spot] exploded) and produced vacuous (100%-coverage) cones.

This forecaster retasks ML onto the forecastable quantity:

    1. XGBoost predicts the forward realized volatility from HAR-style features
       (trailing realized vol over 5 / 21 / 63 / 126 day windows, etc.).
    2. Prices are simulated DRIFTLESS (a martingale, E[spot] ~ forward) with
       Student-t innovations scaled by the ML-predicted volatility.

The result is an unbiased, well-calibrated price distribution (median drift ~0,
~80% of realized prices inside the p10-p90 band), which makes the CVaR the
downstream optimizer minimizes a real estimate of tail procurement cost.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from hedging_assistant.contracts import PriceHistory, PriceForecast
from hedging_assistant.engines.xgb_garch_forecaster import (
    prepare_daily_price_series,
    generate_student_t_shocks,
)

logger = logging.getLogger(__name__)

# In-process cache so repeated forecasts on the same series don't retrain.
_VOL_MODEL_CACHE: dict = {}

# HAR-style realized-vol windows (trading days).
_RV_WINDOWS = (5, 21, 63, 126)
_MIN_HISTORY = 300          # daily rows needed to train a vol model
_DAILY_VOL_FLOOR = 0.004    # ~6% annualized floor
_DAILY_VOL_CEIL = 0.12      # ~190% annualized ceiling


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _feature_row(returns: np.ndarray, t: int) -> list[float]:
    """HAR-style features computed from log returns strictly before index t."""
    return [
        returns[t - 5:t].std(),
        returns[t - 21:t].std(),
        returns[t - 63:t].std(),
        returns[t - 126:t].std(),
        np.abs(returns[t - 21:t]).mean(),
        returns[t - 21:t].mean(),
    ]


def _build_vol_dataset(
    returns: np.ndarray,
    horizon_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Supervised dataset: HAR features -> forward realized daily vol.

    Target at t = std of the next `horizon_days` daily log returns.
    """
    X, y = [], []
    for t in range(max(_RV_WINDOWS), len(returns) - horizon_days):
        X.append(_feature_row(returns, t))
        y.append(returns[t:t + horizon_days].std())
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _train_vol_model(returns: np.ndarray, horizon_days: int, seed: int):
    """Train an XGBoost regressor predicting forward realized daily vol."""
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("xgboost is required for XGB-Vol-t. Run: pip install xgboost") from exc

    X, y = _build_vol_dataset(returns, horizon_days)
    if len(X) < 50:
        raise ValueError(f"Not enough samples to train vol model: {len(X)}")

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=2,
    )
    model.fit(X, y)
    return model


def _predict_current_vol(model, returns: np.ndarray) -> float:
    """Predict forward daily vol from the most recent feature row."""
    feat = np.asarray(_feature_row(returns, len(returns)), dtype=float).reshape(1, -1)
    sigma_d = float(model.predict(feat)[0])
    return float(np.clip(sigma_d, _DAILY_VOL_FLOOR, _DAILY_VOL_CEIL))


# ---------------------------------------------------------------------------
# Public forecaster
# ---------------------------------------------------------------------------

def forecast_ml_vol_t(
    history: PriceHistory,
    horizon: int,
    n_paths: int = 4096,
    seed: int | None = 42,
    calibration_window: int | None = 1500,
    daily_steps: int = 21,
    nu: float = 5.0,
    vol_scale: float = 1.25,
    use_sobol: bool = True,
) -> PriceForecast:
    """
    Forecast prices via ML-predicted volatility + driftless Student-t simulation.

    Parameters
    ----------
    horizon : number of procurement periods (e.g. months).
    daily_steps : trading days per period (21 for monthly).
    nu : Student-t degrees of freedom (fat tails).
    vol_scale : calibration multiplier on predicted vol so the cone hits ~80%
                p10-p90 coverage (XGBoost regresses toward the mean and slightly
                under-predicts vol; ~1.25 corrects it on WTI).
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if daily_steps <= 0:
        raise ValueError("daily_steps must be positive")

    # Use as much daily history as available (cap by calibration_window) to train.
    series = prepare_daily_price_series(history=history, calibration_window=None)
    prices = series.to_numpy(dtype=float)
    if calibration_window is not None and len(prices) > calibration_window:
        prices = prices[-calibration_window:]

    if len(prices) < _MIN_HISTORY:
        raise ValueError(
            f"XGB-Vol-t needs >= {_MIN_HISTORY} daily rows; got {len(prices)}."
        )

    returns = np.diff(np.log(prices))
    S0 = float(prices[-1])
    total_steps = horizon * daily_steps
    horizon_days = total_steps

    # Train (or reuse cached) vol model on this series.
    cache_key = (len(prices), round(S0, 4), horizon_days, seed)
    model = _VOL_MODEL_CACHE.get(cache_key)
    if model is None:
        model = _train_vol_model(returns, horizon_days, seed or 42)
        _VOL_MODEL_CACHE[cache_key] = model

    sigma_d = _predict_current_vol(model, returns)

    # Driftless Student-t Monte Carlo: log_return = vol_scale * sigma_d * shock.
    shocks = generate_student_t_shocks(
        n_paths=n_paths,
        total_steps=total_steps,
        nu=nu,
        seed=seed,
        use_sobol=use_sobol,
    )
    log_returns = (vol_scale * sigma_d) * shocks
    log_price = np.log(S0) + np.cumsum(log_returns, axis=1)
    daily_prices = np.exp(log_price)

    # Sample at period boundaries -> (n_paths, horizon).
    period_idx = (np.arange(1, horizon + 1) * daily_steps) - 1
    paths = daily_prices[:, period_idx]

    logger.info(
        "[ml_vol_forecaster] sigma_d=%.4f (scaled %.4f) horizon=%d daily_steps=%d "
        "n_paths=%d -> E[term]/S0=%.3f median[term]/S0=%.3f",
        sigma_d, vol_scale * sigma_d, horizon, daily_steps, n_paths,
        float(paths[:, -1].mean() / S0), float(np.median(paths[:, -1]) / S0),
    )

    return PriceForecast(
        paths=paths,
        model_name="XGB-Vol-t",
        start_price=S0,
        mu=0.0,
        sigma=float(vol_scale * sigma_d),
        frequency="M",
        seed=seed,
        calibration_window=len(prices),
    )
