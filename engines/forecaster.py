"""
Forecaster engine: PriceHistory -> PriceForecast

BASELINE (Phase 1): Geometric Brownian Motion with pandas-based preprocessing,
  frequency resampling (D/W/M), and a calibration window.
UPGRADE  (Phase 3): GARCH-t (arch lib) or probabilistic ML (Darts/AutoGluon).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from hedging_assistant.contracts import PriceHistory, PriceForecast


def prepare_price_series(
    history: PriceHistory,
    frequency: str = "D",
    calibration_window: int | None = None,
) -> pd.Series:
    """
    Clean and resample raw price history into a pandas Series.

    Args:
        history: raw PriceHistory from the loader
        frequency: "D" daily, "W" weekly, "M" monthly
        calibration_window: if set, keep only the last N rows after resampling

    Returns:
        pd.Series of prices indexed by datetime, sorted ascending, no NaNs
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
    s = s[s > 0]           # drop zero / negative prices (bad EIA rows)
    s = s.dropna()

    if frequency == "W":
        s = s.resample("W-FRI").last().dropna()
    elif frequency == "M":
        s = s.resample("ME").last().dropna()
    # "D" keeps the series as-is (already daily business days)

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


def forecast(
    history: PriceHistory,
    horizon: int,
    n_paths: int = 10_000,
    seed: int | None = 42,
    frequency: str = "D",
    calibration_window: int | None = None,
) -> PriceForecast:
    """
    GBM forecaster.

    Args:
        history: historical price data
        horizon: number of future periods to simulate
        n_paths: number of Monte Carlo paths
        seed: RNG seed for reproducibility (None = random)
        frequency: "D" daily, "W" weekly, "M" monthly — controls resampling
                   AND what one 'period' means in the output paths
        calibration_window: rows of (resampled) history used to fit drift/vol;
                            None = use all available history

    Returns:
        PriceForecast with paths shape (n_paths, horizon)
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1; got {n_paths}")

    series = prepare_price_series(history, frequency, calibration_window)

    log_ret = np.diff(np.log(series.values))
    mu = float(log_ret.mean())
    sigma = float(log_ret.std(ddof=1))

    if sigma <= 0:
        raise ValueError("Computed volatility is zero — price series has no variation.")

    s0 = float(series.iloc[-1])
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, horizon))
    increments = (mu - 0.5 * sigma ** 2) + sigma * z
    paths = s0 * np.exp(np.cumsum(increments, axis=1))

    return PriceForecast(
        paths=paths,
        model_name="GBM",
        frequency=frequency,
        calibration_window=calibration_window,
    )
