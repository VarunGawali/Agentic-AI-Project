"""
Feature engineering utilities for XGB-GARCH-t forecaster.

This module is the single source of truth for drift-model features.

Used by:
    - scripts/train_xgb_drift_model.py
    - engines/xgb_garch_forecaster.py
    - upgraded walk-forward backtester

Feature set:
    - sin_month
    - cos_month
    - log_ret_lag1
    - log_ret_lag5
    - log_ret_lag21
    - rv_5d
    - rv_21d
    - price_dev_60d
    - rsi_14
    - eia_inventory_chg

Target for training:
    y_t = log(P[t+1] / P[t])
"""

from __future__ import annotations
import logging

from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "sin_month",
    "cos_month",
    "log_ret_lag1",
    "log_ret_lag5",
    "log_ret_lag21",
    "rv_5d",
    "rv_21d",
    "price_dev_60d",
    "rsi_14",
    "eia_inventory_chg",
]


# ---------------------------------------------------------------------------
# Basic technical indicators
# ---------------------------------------------------------------------------

def compute_rsi(
    prices: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """
    Compute RSI centered around 0.

    Standard RSI range:
        0 to 100

    Returned range:
        -50 to +50
    """

    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 1:
        raise ValueError("prices must be a 1D array")

    if len(prices) == 0:
        return np.array([], dtype=float)

    if len(prices) <= window:
        return np.zeros(len(prices), dtype=float)

    deltas = np.diff(prices, prepend=prices[0])

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.full(len(prices), np.nan, dtype=float)
    avg_loss = np.full(len(prices), np.nan, dtype=float)

    avg_gain[window] = gains[1: window + 1].mean()
    avg_loss[window] = losses[1: window + 1].mean()

    for i in range(window + 1, len(prices)):
        avg_gain[i] = ((avg_gain[i - 1] * (window - 1)) + gains[i]) / window
        avg_loss[i] = ((avg_loss[i - 1] * (window - 1)) + losses[i]) / window

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)

    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0, posinf=100.0, neginf=0.0)

    return rsi - 50.0


def fill_nan_with_median(values: np.ndarray) -> np.ndarray:
    """
    Replace NaN/inf values with the finite median.
    If no finite values exist, return zeros.
    """

    values = np.asarray(values, dtype=float)

    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        return np.zeros_like(values, dtype=float)

    median = float(np.median(finite))

    return np.nan_to_num(
        values,
        nan=median,
        posinf=median,
        neginf=median,
    )


# ---------------------------------------------------------------------------
# Optional external feature
# ---------------------------------------------------------------------------

def load_inventory_feature(
    inventory_csv: str | Path | None,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Load optional EIA inventory-change feature.

    Expected CSV formats supported:

        date,eia_inventory_chg

    or:

        date,inventory_chg

    or:

        date,value

    If no inventory file is provided, returns zeros.

    In live simulation:
        eia_inventory_chg should usually be set to 0.0 because future inventory
        changes are unknown unless passed as a scenario.
    """

    if inventory_csv is None:
        return np.zeros(len(dates), dtype=float)

    path = Path(inventory_csv)

    if not path.exists():
        logger.warning("[features] Inventory CSV not found: %s. Using zeros.", path)
        return np.zeros(len(dates), dtype=float)

    inventory_df = pd.read_csv(path)

    if "date" not in inventory_df.columns:
        logger.warning("[features] Inventory CSV missing 'date'. Using zeros.")
        return np.zeros(len(dates), dtype=float)

    possible_value_columns = [
        "eia_inventory_chg",
        "inventory_chg",
        "value",
    ]

    value_col = None

    for col in possible_value_columns:
        if col in inventory_df.columns:
            value_col = col
            break

    if value_col is None:
        logger.warning("[features] Inventory CSV missing inventory value column. Using zeros.")
        return np.zeros(len(dates), dtype=float)

    inventory_df["date"] = pd.to_datetime(inventory_df["date"], errors="coerce")
    inventory_df[value_col] = pd.to_numeric(
        inventory_df[value_col],
        errors="coerce",
    )

    inventory_df = inventory_df.dropna(subset=["date", value_col])
    inventory_df = inventory_df.sort_values("date")
    inventory_df = inventory_df.drop_duplicates("date", keep="last")

    inventory_series = inventory_df.set_index("date")[value_col]

    aligned = inventory_series.reindex(dates).ffill().fillna(0.0)

    return aligned.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_features(
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    eia_inventory_chg: np.ndarray | None = None,
) -> pd.DataFrame:

    """
    Build model features from a full historical price series.

    X[t] predicts:

        y[t] = log(P[t+1] / P[t])

    Parameters:
        prices:
            Positive price series, shape (T,).

        dates:
            DatetimeIndex, shape (T,).

        eia_inventory_chg:
            Optional external feature, shape (T,).
            If missing, zeros are used.


    Returns:
        DataFrame with FEATURE_COLUMNS.
    """

    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 1:
        raise ValueError("prices must be a 1D array")

    if len(prices) == 0:
        raise ValueError("prices cannot be empty")

    if np.any(prices <= 0):
        raise ValueError("prices must be positive")

    if len(prices) != len(dates):
        raise ValueError("prices and dates must have the same length")

    dates = pd.DatetimeIndex(dates)

    log_prices = np.log(prices)
    log_returns = np.diff(log_prices, prepend=log_prices[0])

    # -----------------------------------------------------------------------
    # Seasonality
    # -----------------------------------------------------------------------

    month = dates.month.to_numpy(dtype=float)

    sin_month = np.sin(2.0 * np.pi * month / 12.0)
    cos_month = np.cos(2.0 * np.pi * month / 12.0)

    # -----------------------------------------------------------------------
    # Momentum / return lags
    # -----------------------------------------------------------------------

    log_ret_lag1 = np.concatenate([[0.0], log_returns[:-1]])
    log_ret_lag5 = np.concatenate([np.zeros(5), log_returns[:-5]])
    log_ret_lag21 = np.concatenate([np.zeros(21), log_returns[:-21]])

    # -----------------------------------------------------------------------
    # Realized volatility
    # -----------------------------------------------------------------------

    ret_series = pd.Series(log_returns)

    rv_5d = (
        ret_series
        .rolling(5, min_periods=2)
        .std()
        .to_numpy()
        * np.sqrt(252)
    )

    rv_21d = (
        ret_series
        .rolling(21, min_periods=2)
        .std()
        .to_numpy()
        * np.sqrt(252)
    )

    rv_5d = fill_nan_with_median(rv_5d)
    rv_21d = fill_nan_with_median(rv_21d)

    # -----------------------------------------------------------------------
    # Price deviation from 60-day moving average
    # -----------------------------------------------------------------------

    ma_60d = (
        pd.Series(prices)
        .rolling(60, min_periods=10)
        .mean()
        .to_numpy()
    )

    ma_60d = np.where(np.isnan(ma_60d), prices, ma_60d)
    price_dev_60d = np.log(prices / np.maximum(ma_60d, 1e-8))

    # -----------------------------------------------------------------------
    # RSI
    # -----------------------------------------------------------------------

    rsi_14 = compute_rsi(prices, window=14)

    # -----------------------------------------------------------------------
    # External inventory feature
    # -----------------------------------------------------------------------

    if eia_inventory_chg is None or len(eia_inventory_chg) != len(prices):
        eia_inventory_chg = np.zeros(len(prices), dtype=float)
    else:
        eia_inventory_chg = np.asarray(eia_inventory_chg, dtype=float)

    features = pd.DataFrame(
        {
            "sin_month": sin_month,
            "cos_month": cos_month,
            "log_ret_lag1": log_ret_lag1,
            "log_ret_lag5": log_ret_lag5,
            "log_ret_lag21": log_ret_lag21,
            "rv_5d": rv_5d,
            "rv_21d": rv_21d,
            "price_dev_60d": price_dev_60d,
            "rsi_14": rsi_14,
            "eia_inventory_chg": eia_inventory_chg,
        }
    )

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0.0)

    return features[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Supervised dataset builder for training
# ---------------------------------------------------------------------------

def make_supervised_dataset(
    df: pd.DataFrame,
    inventory_csv: str | Path | None = None,
    warmup: int = 60,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Convert price history DataFrame into XGBoost supervised dataset.

    Input DataFrame requires:
        date
        price

    Target:
        y[t] = log(P[t+1] / P[t])

    Returns:
        X:
            Feature matrix.

        y:
            Next-period log return.

        target_df:
            DataFrame aligned with y after warmup.
            Useful for metadata/backtesting.
    """

    required_columns = {"date", "price"}

    if not required_columns.issubset(df.columns):
        raise ValueError("Input data must contain 'date' and 'price' columns")

    clean_df = df.copy()

    clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
    clean_df["price"] = pd.to_numeric(clean_df["price"], errors="coerce")

    clean_df = clean_df.dropna(subset=["date", "price"])
    clean_df = clean_df[clean_df["price"] > 0]
    clean_df = clean_df.sort_values("date")
    clean_df = clean_df.drop_duplicates("date", keep="last")
    clean_df = clean_df.reset_index(drop=True)

    min_required = max(150, warmup + 30)

    if len(clean_df) < min_required:
        raise ValueError(
            f"Need at least {min_required} rows to build supervised dataset; "
            f"got {len(clean_df)}"
        )

    dates = pd.DatetimeIndex(clean_df["date"])
    prices = clean_df["price"].to_numpy(dtype=float)

    inventory = load_inventory_feature(
        inventory_csv=inventory_csv,
        dates=dates,
    )

    features = build_features(
        prices=prices,
        dates=dates,
        eia_inventory_chg=inventory,
    )

    log_prices = np.log(prices)
    target = np.diff(log_prices)

    # X[t] predicts log(P[t+1] / P[t])
    X = features.iloc[:-1].copy()
    y = target.copy()

    if warmup > 0:
        X = X.iloc[warmup:].reset_index(drop=True)
        y = y[warmup:]

        # y after warmup corresponds to target date t+1
        target_df = clean_df.iloc[warmup + 1:].reset_index(drop=True)
    else:
        X = X.reset_index(drop=True)
        target_df = clean_df.iloc[1:].reset_index(drop=True)

    if len(X) != len(y):
        raise ValueError(
            f"Feature/target length mismatch: len(X)={len(X)}, len(y)={len(y)}"
        )

    if len(target_df) != len(y):
        raise ValueError(
            f"Target metadata length mismatch: len(target_df)={len(target_df)}, "
            f"len(y)={len(y)}"
        )

    return X, y, target_df


# ---------------------------------------------------------------------------
# Latest feature row for inference
# ---------------------------------------------------------------------------

def build_latest_feature_row(
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    eia_inventory_chg: float = 0.0,
) -> pd.DataFrame:
    """
    Build the latest feature row for live inference.

    Used when predicting the next-period drift from the trained XGBoost model.

    For future simulation:
        eia_inventory_chg should usually be 0.0 unless scenario-driven.
    """

    prices = np.asarray(prices, dtype=float)
    dates = pd.DatetimeIndex(dates)

    if len(prices) != len(dates):
        raise ValueError("prices and dates must have the same length")

    inventory_arr = np.zeros(len(prices), dtype=float)
    inventory_arr[-1] = float(eia_inventory_chg)

    features = build_features(
        prices=prices,
        dates=dates,
        eia_inventory_chg=inventory_arr,
    )

    return features.iloc[[-1]][FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Simulation feature row
# ---------------------------------------------------------------------------

def build_simulation_feature_row(
    price_buffer: list[float] | np.ndarray,
    current_date: pd.Timestamp,
    long_run_vol: float,
    eia_inventory_chg: float = 0.0,
) -> pd.DataFrame:
    """
    Build one feature row during forward simulation.

    This function is used inside the XGB-GARCH-t simulator.

    price_buffer:
        Recent simulated + historical prices.
        Ideally contains at least 60 observations.

    current_date:
        Date corresponding to the current simulated step.

    long_run_vol:
        Fallback volatility value used when the buffer is too short.
    """

    prices = np.asarray(price_buffer, dtype=float)

    if prices.ndim != 1:
        raise ValueError("price_buffer must be 1D")

    if len(prices) == 0:
        raise ValueError("price_buffer cannot be empty")

    if np.any(prices <= 0):
        raise ValueError("price_buffer must contain positive prices")

    current_date = pd.Timestamp(current_date)

    # Create synthetic dates ending at current_date.
    # Business-day spacing is enough for feature construction.
    dates = pd.bdate_range(
        end=current_date,
        periods=len(prices),
    )

    features = build_features(
        prices=prices,
        dates=dates,
        eia_inventory_chg=np.full(len(prices), float(eia_inventory_chg)),
    )

    latest = features.iloc[[-1]].copy()

    # If buffer is too short and rolling vol collapsed to zero,
    # stabilize with long_run_vol.
    if len(prices) < 21:
        latest["rv_5d"] = float(long_run_vol)
        latest["rv_21d"] = float(long_run_vol)

    return latest[FEATURE_COLUMNS]