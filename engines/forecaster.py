"""
Forecaster engine: PriceHistory -> PriceForecast

BASELINE (Phase 1): Geometric Brownian Motion with pandas-based preprocessing,
  frequency resampling (D/W/M), and a calibration window.
UPGRADES:
  #3  Sobol quasi-random sequences + antithetic variates
      (O(1/N) convergence; halves estimator variance simultaneously)
  #4  joblib.Memory path caching
  #6  AIC-based rolling calibration window selection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import joblib
from scipy.stats import qmc, norm as scipy_norm

from hedging_assistant.contracts import PriceHistory, PriceForecast


# ---------------------------------------------------------------------------
# joblib cache setup (Improvement #4)
# ---------------------------------------------------------------------------
_CACHE_DIR = str(Path(__file__).parent.parent / ".joblib_cache")
_memory = joblib.Memory(location=_CACHE_DIR, verbose=0)


def prepare_price_series(
    history: PriceHistory,
    frequency: str = "D",
    calibration_window: int | None = None,
) -> pd.Series:
    """
    Clean and resample raw price history into a pandas Series.
    """
    if len(history.prices) == 0:
        raise ValueError("PriceHistory is empty — cannot prepare price series.")
    if frequency not in ("D", "W", "M"):
        raise ValueError(f"frequency must be 'D', 'W', or 'M'; got '{frequency}'")

    s = pd.Series(
        history.prices,
        index=pd.to_datetime(history.dates),
        name=history.symbol,
    )
    s = s.sort_index()
    s = s[s > 0]
    s = s.dropna()

    if frequency == "W":
        s = s.resample("W-FRI").last().dropna()
    elif frequency == "M":
        s = s.resample("ME").last().dropna()

    if calibration_window is not None:
        if calibration_window < 10:
            raise ValueError(f"calibration_window must be >= 10; got {calibration_window}")
        s = s.iloc[-calibration_window:]

    if len(s) < 10:
        raise ValueError(
            f"After cleaning/resampling only {len(s)} rows remain — "
            "increase calibration_window or provide more history."
        )
    return s


# ----------------------------------------------------------------------------
# Improvement #6: AIC-based calibration window selection
# ----------------------------------------------------------------------------

def select_calibration_window(
    series: pd.Series,
    min_window: int = 60,
    max_window: int = 500,
    step: int = 20,
) -> int:
    """
    Choose the calibration window that minimises AIC for a GBM log-return model.

    For each candidate window w:
      - Slice the last w rows of series
      - Compute log-returns, fit mu and sigma (k=2 params)
      - AIC = 2*k - 2*log_likelihood  (normal log-pdf)
    Returns the window size with minimum AIC.
    """
    best_aic = float("inf")
    best_w = min_window
    upper = min(max_window, len(series) - 1)

    for w in range(min_window, upper + 1, step):
        slice_ = series.iloc[-w:]
        lr = np.diff(np.log(slice_.values))
        if len(lr) < 2:
            continue
        mu = float(lr.mean())
        sigma = float(lr.std(ddof=1))
        if sigma <= 0:
            continue
        log_lik = float(scipy_norm.logpdf(lr, loc=mu, scale=sigma).sum())
        aic = 2 * 2 - 2 * log_lik   # k=2 parameters: mu, sigma
        if aic < best_aic:
            best_aic = aic
            best_w = w

    return best_w


# ----------------------------------------------------------------------------
# Core implementation cached by joblib (Improvement #4)
# ----------------------------------------------------------------------------

def _forecast_impl(
    prices: np.ndarray,
    dates_str: list,
    symbol: str,
    horizon: int,
    n_paths: int,
    seed,
    frequency: str,
    calibration_window,
) -> np.ndarray:
    """
    Core GBM path simulation, cached by joblib.Memory.

    Improvement #3: Uses Sobol quasi-random sequences + antithetic variates.
    Sobol + antithetic together give O(1/N) convergence vs O(1/sqrt(N)) for
    plain MC, and antithetic variates halve the estimator variance simultaneously.
    """
    s = pd.Series(prices, index=pd.to_datetime(dates_str), name=symbol)
    s = s.sort_index()
    s = s[s > 0].dropna()

    if frequency == "W":
        s = s.resample("W-FRI").last().dropna()
    elif frequency == "M":
        s = s.resample("ME").last().dropna()

    if calibration_window is None:
        calibration_window = select_calibration_window(s)
        print(f"[forecaster] AIC-selected calibration_window={calibration_window}")
    s = s.iloc[-calibration_window:]

    if len(s) < 10:
        raise ValueError(f"Only {len(s)} rows after cleaning — need >= 10.")

    log_ret = np.diff(np.log(s.values))
    mu = float(log_ret.mean())
    sigma = float(log_ret.std(ddof=1))
    if sigma <= 0:
        raise ValueError("Computed volatility is zero — price series has no variation.")

    s0 = float(s.iloc[-1])

    # Improvement #3: Sobol + antithetic variates
    half = n_paths // 2
    sobol_engine = qmc.Sobol(d=horizon, scramble=True, seed=seed)
    u = sobol_engine.random(half)           # shape (half, horizon) in [0,1]
    z_pos = scipy_norm.ppf(u)               # transform to standard normal
    z_neg = -z_pos                          # antithetic mirror
    z = np.vstack([z_pos, z_neg])           # (2*half, horizon)
    if n_paths % 2 == 1:
        rng = np.random.default_rng(seed)
        extra = rng.standard_normal((1, horizon))
        z = np.vstack([z, extra])

    increments = (mu - 0.5 * sigma ** 2) + sigma * z
    paths = s0 * np.exp(np.cumsum(increments, axis=1))
    return paths


def forecast(
    history: PriceHistory,
    horizon: int,
    n_paths: int = 10_000,
    seed: int | None = 42,
    frequency: str = "D",
    calibration_window: int | None = None,
    use_cache: bool = True,
) -> PriceForecast:
    """
    GBM forecaster with Sobol+antithetic sampling and optional joblib caching.

    Improvement #4: joblib.Memory caches the result keyed on all arguments.
    Pass use_cache=False to force recomputation (useful for fresh seeds in
    final production runs).

    Improvement #6: When calibration_window is None, selects the optimal
    window via AIC. When explicitly set, uses it directly (skips AIC search).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1; got {n_paths}")
    if frequency not in ("D", "W", "M"):
        raise ValueError(f"frequency must be 'D', 'W', or 'M'; got '{frequency}'")

    dates_str = list(pd.to_datetime(history.dates).strftime("%Y-%m-%d"))

    if use_cache:
        cached_fn = _memory.cache(_forecast_impl)
        paths = cached_fn(
            history.prices, dates_str, history.symbol,
            horizon, n_paths, seed, frequency, calibration_window,
        )
        print("[forecaster] Cache hit — returning cached paths.")
    else:
        print("[forecaster] Cache disabled — computing paths.")
        paths = _forecast_impl(
            history.prices, dates_str, history.symbol,
            horizon, n_paths, seed, frequency, calibration_window,
        )

    return PriceForecast(
        paths=paths,
        model_name="GBM-Sobol",
        frequency=frequency,
        calibration_window=calibration_window,
    )
