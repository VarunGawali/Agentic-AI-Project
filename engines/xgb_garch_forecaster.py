"""
XGB- design:XGB-GARCH-t forecaster.
    - XGBoost drift model is trained offline by scripts/train_xgb_drift_model.py.
    - This module loads the trained XGBoost model.
    - GARCH(1,1)-t is fitted on recent XGBoost residuals.
    - Future paths are simulated using:

        log_return_t = xgb_drift_t + sigma_t * t_shock_t

Outputs:
    PriceForecast
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import qmc

from contracts import PriceHistory, PriceForecast
from engines.features import (
    FEATURE_COLUMNS,
    build_features,
)


DEFAULT_MODEL_PATH = Path("models/xgb_drift_model.json")
DEFAULT_FEATURE_COLUMNS_PATH = Path("models/xgb_feature_columns.json")


# ---------------------------------------------------------------------------
# Price preparation
# ---------------------------------------------------------------------------

def prepare_daily_price_series(
    history: PriceHistory,
    calibration_window: int | None = None,
) -> pd.Series:
    """
    Clean raw price history into daily price series.

    XGB-GARCH-t works on daily data.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(history.dates),
            "price": history.prices,
        }
    )

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]
    df = df.sort_values("date")
    df = df.drop_duplicates("date", keep="last")

    if df.empty:
        raise ValueError("No valid price data available for XGB-GARCH-t.")

    series = df.set_index("date")["price"]

    if calibration_window is not None:
        if calibration_window < 150:
            raise ValueError(
                "calibration_window must be at least 150 for XGB-GARCH-t."
            )

        if len(series) < calibration_window:
            raise ValueError(
                f"Not enough rows for calibration_window={calibration_window}. "
                f"Got {len(series)} rows."
            )

        series = series.iloc[-calibration_window:]

    if len(series) < 150:
        raise ValueError(
            f"Need at least 150 daily rows for XGB-GARCH-t; got {len(series)}."
        )

    return series


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_xgb_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
):
    """
    Load trained XGBoost drift model.
    """

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for XGB-GARCH-t. Run: uv pip install xgboost"
        ) from exc

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"XGBoost model artifact not found: {model_path}. "
            "Run: uv run python -m scripts.train_xgb_drift_model"
        )

    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    return model


def load_feature_columns(
    feature_columns_path: str | Path = DEFAULT_FEATURE_COLUMNS_PATH,
) -> list[str]:
    """
    Load feature column order used during XGBoost training.
    """

    feature_columns_path = Path(feature_columns_path)

    if not feature_columns_path.exists():
        raise FileNotFoundError(
            f"Feature column artifact not found: {feature_columns_path}. "
            "Run: uv run python -m scripts.train_xgb_drift_model"
        )

    with open(feature_columns_path, "r", encoding="utf-8") as f:
        columns = json.load(f)

    if columns != FEATURE_COLUMNS:
        raise ValueError(
            "Feature column mismatch between saved model and engines.features.\n"
            f"Saved columns   : {columns}\n"
            f"Current columns : {FEATURE_COLUMNS}"
        )

    return columns


# ---------------------------------------------------------------------------
# Residuals + GARCH fitting
# ---------------------------------------------------------------------------

def build_drift_residuals(
    series: pd.Series,
    xgb_model,
    feature_columns: list[str],
    warmup: int = 60,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Build XGBoost residuals on historical data.

    residual_t = actual_next_log_return_t - predicted_next_log_return_t
    """

    prices = series.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(series.index)

    features = build_features(
        prices=prices,
        dates=dates,
        eia_inventory_chg=np.zeros(len(prices), dtype=float),
    )

    log_prices = np.log(prices)
    target = np.diff(log_prices)

    X = features.iloc[:-1].copy()
    y = target.copy()

    if warmup > 0:
        X = X.iloc[warmup:].reset_index(drop=True)
        y = y[warmup:]

    X = X[feature_columns]

    if len(X) != len(y):
        raise ValueError("Feature/target length mismatch while building residuals.")

    predictions = xgb_model.predict(X)
    residuals = y - predictions

    residuals = np.asarray(residuals, dtype=float)

    if len(residuals) < 50:
        raise ValueError(
            f"Need at least 50 residuals to fit GARCH; got {len(residuals)}."
        )

    return residuals, X, predictions


def fit_garch_t_on_residuals(
    residuals: np.ndarray,
) -> dict:
    """
    Fit GARCH(1,1)-t on XGBoost residuals.

    Uses percent units internally because arch package is usually more stable
    when financial returns are expressed in percentage points.
    """

    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError(
            "arch is required for XGB-GARCH-t. Run: uv pip install arch"
        ) from exc

    residuals = np.asarray(residuals, dtype=float)

    if residuals.ndim != 1:
        raise ValueError("residuals must be 1D")

    if len(residuals) < 50:
        raise ValueError("Need at least 50 residuals to fit GARCH-t")

    residuals_pct = residuals * 100.0

    model = arch_model(
        residuals_pct,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )

    result = model.fit(
        disp="off",
        show_warning=False,
    )

    params = result.params

    omega = float(params.get("omega", 0.01))
    alpha = float(params.get("alpha[1]", 0.05))
    beta = float(params.get("beta[1]", 0.90))
    nu = float(params.get("nu", 8.0))

    nu = max(nu, 2.5)

    conditional_vol_pct = np.asarray(result.conditional_volatility, dtype=float)

    last_sigma2_pct = float(conditional_vol_pct[-1] ** 2)
    last_residual_pct = float(residuals_pct[-1])

    persistence = alpha + beta

    if persistence < 1.0:
        denominator = max(1.0 - persistence, 1e-8)
        long_run_sigma_pct = float(np.sqrt(omega / denominator))
    else:
        long_run_sigma_pct = float(np.sqrt(last_sigma2_pct))

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "nu": nu,
        "last_sigma2_pct": last_sigma2_pct,
        "last_residual_pct": last_residual_pct,
        "long_run_vol": long_run_sigma_pct / 100.0,
        "persistence": persistence,
    }


# ---------------------------------------------------------------------------
# Shock generation
# ---------------------------------------------------------------------------

def generate_student_t_shocks(
    n_paths: int,
    total_steps: int,
    nu: float,
    seed: int | None = 42,
    use_sobol: bool = True,
) -> np.ndarray:
    """
    Generate standardized Student-t shocks.

    Uses Sobol + antithetic variates when use_sobol=True.
    """

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if nu <= 2:
        nu = 2.5

    if not use_sobol:
        rng = np.random.default_rng(seed)
        shocks = rng.standard_t(df=nu, size=(n_paths, total_steps))
    else:
        half = n_paths // 2

        if half > 0:
            m = int(np.ceil(np.log2(half)))

            sobol = qmc.Sobol(
                d=total_steps,
                scramble=True,
                seed=seed,
            )

            u_full = sobol.random_base2(m=m)
            u = u_full[:half]

            eps = np.finfo(float).eps
            u = np.clip(u, eps, 1.0 - eps)

            z_pos = scipy.stats.t.ppf(u, df=nu)
            z_neg = -z_pos

            shocks = np.vstack([z_pos, z_neg])
        else:
            shocks = np.empty((0, total_steps))

        if n_paths % 2 == 1:
            rng = np.random.default_rng(seed)
            extra = rng.standard_t(df=nu, size=(1, total_steps))
            shocks = np.vstack([shocks, extra])

    # Standardize t innovations to unit variance.
    theoretical_std = np.sqrt(nu / (nu - 2.0))

    if theoretical_std > 0 and np.isfinite(theoretical_std):
        shocks = shocks / theoretical_std

    return shocks


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _build_batch_simulation_features(
    price_buffer: np.ndarray,
    current_date: pd.Timestamp,
    long_run_vol: float,
) -> pd.DataFrame:
    """
    Build one simulation feature row per path.

    IMPORTANT:
        This must match engines.features.build_features() definitions.

    Training definitions:
        log_ret_lag1  = previous daily log return
        log_ret_lag5  = daily log return from 5 periods ago
        log_ret_lag21 = daily log return from 21 periods ago

    price_buffer:
        shape = (n_paths, buffer_len)

    Returns:
        DataFrame shape = (n_paths, len(FEATURE_COLUMNS))
    """

    price_buffer = np.asarray(price_buffer, dtype=float)

    if price_buffer.ndim != 2:
        raise ValueError("price_buffer must be 2D: (n_paths, buffer_len)")

    if np.any(price_buffer <= 0):
        raise ValueError("price_buffer must contain positive prices")

    n_paths, buffer_len = price_buffer.shape
    current_date = pd.Timestamp(current_date)

    month = float(current_date.month)

    sin_month = np.full(
        n_paths,
        np.sin(2.0 * np.pi * month / 12.0),
        dtype=float,
    )

    cos_month = np.full(
        n_paths,
        np.cos(2.0 * np.pi * month / 12.0),
        dtype=float,
    )

    current_price = price_buffer[:, -1]
    log_buffer = np.log(price_buffer)

    # Match build_features():
    # log_returns[t] = log(P_t / P_{t-1})
    # log_returns[0] = 0
    log_returns = np.zeros_like(log_buffer)
    log_returns[:, 1:] = np.diff(log_buffer, axis=1)

    def lagged_daily_return(lag: int) -> np.ndarray:
        """
        Match training:
            lag1[t]  = log_returns[t-1]
            lag5[t]  = log_returns[t-5]
            lag21[t] = log_returns[t-21]
        """

        idx = buffer_len - 1 - lag

        if idx < 0:
            return np.zeros(n_paths, dtype=float)

        return log_returns[:, idx]

    log_ret_lag1 = lagged_daily_return(1)
    log_ret_lag5 = lagged_daily_return(5)
    log_ret_lag21 = lagged_daily_return(21)

    def realized_vol(window: int) -> np.ndarray:
        """
        Match training rolling std over daily log returns.
        """

        if buffer_len < max(2, window):
            return np.full(n_paths, long_run_vol, dtype=float)

        sliced = log_returns[:, -window:]

        vol = np.std(sliced, axis=1, ddof=1) * np.sqrt(252)

        vol = np.nan_to_num(
            vol,
            nan=long_run_vol,
            posinf=long_run_vol,
            neginf=long_run_vol,
        )

        vol = np.where(vol <= 0, long_run_vol, vol)

        return vol

    rv_5d = realized_vol(5)
    rv_21d = realized_vol(21)

    ma_60d = price_buffer[:, -60:].mean(axis=1)
    price_dev_60d = np.log(current_price / np.maximum(ma_60d, 1e-8))

    rsi_14 = _batch_rsi_centered(price_buffer[:, -30:])

    eia_inventory_chg = np.zeros(n_paths, dtype=float)

    return pd.DataFrame(
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



def _batch_rsi_centered(
    price_windows: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """
    Approximate centered RSI for many paths using recent price windows.

    Returns:
        RSI - 50, shape = (n_paths,)
    """

    price_windows = np.asarray(price_windows, dtype=float)

    if price_windows.ndim != 2:
        raise ValueError("price_windows must be 2D")

    n_paths, width = price_windows.shape

    if width < window + 1:
        return np.zeros(n_paths, dtype=float)

    recent = price_windows[:, -window - 1:]

    deltas = np.diff(recent, axis=1)

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains.mean(axis=1)
    avg_loss = losses.mean(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)

    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0, posinf=100.0, neginf=0.0)

    return rsi - 50.0


def _next_business_day(
    date: pd.Timestamp,
) -> pd.Timestamp:
    """
    Return next business day after given date.
    """

    next_date = pd.Timestamp(date) + pd.Timedelta(days=1)

    while next_date.weekday() >= 5:
        next_date += pd.Timedelta(days=1)

    return next_date


def simulate_xgb_garch_paths(
    series: pd.Series,
    xgb_model,
    feature_columns: list[str],
    garch_params: dict,
    horizon: int,
    n_paths: int,
    seed: int | None = 42,
    daily_steps: int = 21,
    use_sobol: bool = True,
    drift_scale: float = 0.25,
    drift_clip: float = 0.01,
    debug: bool = False,
) -> np.ndarray:
    """
    Simulate monthly forecast paths using XGBoost drift + GARCH-t volatility.

    Optimized version:
        - Builds features for all paths at once per simulated day.
        - Calls XGBoost once per day, not once per path.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    if daily_steps <= 0:
        raise ValueError("daily_steps must be positive")

    prices = series.to_numpy(dtype=float)

    if np.any(prices <= 0):
        raise ValueError("series must contain positive prices")

    last_price = float(prices[-1])
    last_date = pd.Timestamp(series.index[-1])

    total_steps = horizon * daily_steps

    omega = float(garch_params["omega"])
    alpha = float(garch_params["alpha"])
    beta = float(garch_params["beta"])
    nu = float(garch_params["nu"])
    long_run_vol = float(garch_params["long_run_vol"])

    shocks = generate_student_t_shocks(
        n_paths=n_paths,
        total_steps=total_steps,
        nu=nu,
        seed=seed,
        use_sobol=use_sobol,
    )

    output_paths = np.empty((n_paths, horizon), dtype=float)

    # Warm-start feature buffer with last 125 historical prices.
    history_buffer = prices[-125:]

    if len(history_buffer) < 125:
        pad_value = history_buffer[0]
        pad = np.full(125 - len(history_buffer), pad_value, dtype=float)
        history_buffer = np.concatenate([pad, history_buffer])

    # shape: (n_paths, 125)
    price_buffer = np.tile(history_buffer[None, :], (n_paths, 1))

    log_price = np.full(n_paths, np.log(last_price), dtype=float)

    sigma2_pct = np.full(
        n_paths,
        float(garch_params["last_sigma2_pct"]),
        dtype=float,
    )

    last_residual_pct = np.full(
        n_paths,
        float(garch_params["last_residual_pct"]),
        dtype=float,
    )

    sim_date = last_date
    month_idx = 0
    day_in_period = 0

    for step in range(total_steps):
        sim_date = _next_business_day(sim_date)

        feature_frame = _build_batch_simulation_features(
            price_buffer=price_buffer,
            current_date=sim_date,
            long_run_vol=long_run_vol,
        )

        feature_frame = feature_frame[feature_columns]

        # One XGBoost call for all paths.
        # One XGBoost call for all paths.
        drift = xgb_model.predict(feature_frame).astype(float)

        # Guardrail:
        # XGBoost is used as a weak conditional drift estimator.
        # GARCH-t should drive most of the stochastic variation.
        drift = drift * drift_scale
        drift = np.clip(drift, -drift_clip, drift_clip)

        if debug and step in {0, 20, 60, 120}:
            print(
                f"[xgb-garch-debug] step={step}, "
                f"drift_mean={drift.mean():.5f}, "
                f"drift_p10={np.percentile(drift, 10):.5f}, "
                f"drift_p90={np.percentile(drift, 90):.5f}, "
                f"sigma_mean={(np.sqrt(sigma2_pct).mean() / 100.0):.5f}"
            )

        # GARCH update in percent units.
        sigma2_pct = (
            omega
            + alpha * (last_residual_pct ** 2)
            + beta * sigma2_pct
        )

        sigma2_pct = np.maximum(sigma2_pct, 1e-10)

        sigma_decimal = np.sqrt(sigma2_pct) / 100.0
        z = shocks[:, step]

        simulated_residual = sigma_decimal * z
        log_return = drift + simulated_residual

        log_price = log_price + log_return
        new_price = np.exp(log_price)

        if np.any(~np.isfinite(new_price)) or np.any(new_price <= 0):
            raise ValueError("Invalid simulated price generated.")

        last_residual_pct = simulated_residual * 100.0

        # Roll buffer left and append newest simulated price.
        price_buffer[:, :-1] = price_buffer[:, 1:]
        price_buffer[:, -1] = new_price

        day_in_period += 1

        if day_in_period == daily_steps:
            output_paths[:, month_idx] = new_price
            month_idx += 1
            day_in_period = 0

    return output_paths




# ---------------------------------------------------------------------------
# Public forecaster
# ---------------------------------------------------------------------------

def forecast_xgb_garch_t(
    history: PriceHistory,
    horizon: int,
    n_paths: int = 4096,
    seed: int | None = 42,
    calibration_window: int | None = 1000,
    daily_steps: int = 21,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    feature_columns_path: str | Path = DEFAULT_FEATURE_COLUMNS_PATH,
    use_sobol: bool = True,
    drift_scale: float = 0.25,
    drift_clip: float = 0.01,
    debug: bool = False,
) -> PriceForecast:
    """
    Forecast crude prices using trained XGBoost drift model + GARCH(1,1)-t.

    This function does not train XGBoost.
    It loads the trained model artifact produced by:

        uv run python -m scripts.train_xgb_drift_model
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    series = prepare_daily_price_series(
        history=history,
        calibration_window=calibration_window,
    )

    xgb_model = load_xgb_model(model_path)
    feature_columns = load_feature_columns(feature_columns_path)

    residuals, _, _ = build_drift_residuals(
        series=series,
        xgb_model=xgb_model,
        feature_columns=feature_columns,
        warmup=60,
    )

    garch_params = fit_garch_t_on_residuals(residuals)

    print(
        "[xgb-garch] fitted GARCH-t: "
        f"omega={garch_params['omega']:.4f}, "
        f"alpha={garch_params['alpha']:.4f}, "
        f"beta={garch_params['beta']:.4f}, "
        f"nu={garch_params['nu']:.2f}, "
        f"persistence={garch_params['persistence']:.4f}, "
        f"long_run_vol={garch_params['long_run_vol']:.4f}"
    )

    paths = simulate_xgb_garch_paths(
        series=series,
        xgb_model=xgb_model,
        feature_columns=feature_columns,
        garch_params=garch_params,
        horizon=horizon,
        n_paths=n_paths,
        seed=seed,
        daily_steps=daily_steps,
        use_sobol=use_sobol,
        drift_scale=drift_scale,
        drift_clip=drift_clip,
        debug=debug,
    )

    log_returns = np.diff(np.log(series.to_numpy(dtype=float)))

    return PriceForecast(
        paths=paths,
        model_name="XGB-GARCH-t",
        start_price=float(series.iloc[-1]),
        mu=float(log_returns.mean()),
        sigma=float(log_returns.std(ddof=1)),
        frequency="M",
        seed=seed,
        calibration_window=len(series),
    )

