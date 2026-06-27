"""
Forecaster engine: PriceHistory -> PriceForecast

BASELINE:
    GBM forecaster with D/W/M resampling and calibration window.

IMPROVEMENTS:
    - Sobol quasi-random sampling
    - Antithetic variates
    - Optional joblib caching
    - Optional AIC-based calibration window selection
    - Optional Student-t shocks for fat tails
    - Optional HMM regime-aware mu/sigma override
    - Fan chart plotting

Notes:
    - use_regime=False by default because hmmlearn may not be installed.
    - distribution="normal" by default.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import joblib
from hedging_assistant.engines.xgb_garch_forecaster import forecast_xgb_garch_t
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm as scipy_norm
from scipy.stats import qmc

from hedging_assistant.contracts import PriceHistory, PriceForecast


# ---------------------------------------------------------------------------
# joblib cache setup
# ---------------------------------------------------------------------------

_CACHE_DIR = str(Path(__file__).parent.parent / ".joblib_cache")
_memory = joblib.Memory(location=_CACHE_DIR, verbose=0)


# ---------------------------------------------------------------------------
# Price preparation
# ---------------------------------------------------------------------------

def prepare_price_series(
    history: PriceHistory,
    frequency: str = "D",
) -> pd.Series:
    """
    Clean, sort, deduplicate, and resample crude price history.

    frequency:
        D = daily
        W = weekly
        M = monthly
    """

    frequency = frequency.upper()

    if frequency not in {"D", "W", "M"}:
        raise ValueError(f"frequency must be 'D', 'W', or 'M'; got '{frequency}'")

    if len(history.prices) == 0:
        raise ValueError("PriceHistory is empty.")

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

    series = df.set_index("date")["price"]

    if frequency == "D":
        return series.dropna()

    if frequency == "W":
        return series.resample("W-FRI").last().dropna()

    if frequency == "M":
        return series.resample("ME").last().dropna()

    raise ValueError("frequency must be one of: 'D', 'W', 'M'.")


# ---------------------------------------------------------------------------
# AIC-based calibration window selection
# ---------------------------------------------------------------------------

def select_calibration_window(
    series: pd.Series,
    min_window: int = 60,
    max_window: int = 500,
    step: int = 20,
) -> int:
    """
    Select calibration window using AIC on GBM log returns.

    The returned window is interpreted as number of return observations.
    Example:
        selected_window=60 means use latest 61 prices to compute 60 returns.
    """

    if len(series) < min_window + 1:
        return max(2, len(series) - 1)

    best_aic = float("inf")
    best_w = min_window

    upper = min(max_window, len(series) - 1)

    for w in range(min_window, upper + 1, step):
        price_slice = series.iloc[-(w + 1):]
        log_returns = np.diff(np.log(price_slice.values))

        if len(log_returns) < 2:
            continue

        mu = float(log_returns.mean())
        sigma = float(log_returns.std(ddof=1))

        if np.isnan(mu) or np.isnan(sigma) or sigma <= 0:
            continue

        log_likelihood = float(
            scipy_norm.logpdf(log_returns, loc=mu, scale=sigma).sum()
        )

        k = 2
        aic = 2 * k - 2 * log_likelihood

        if aic < best_aic:
            best_aic = aic
            best_w = w

    return best_w


# ---------------------------------------------------------------------------
# Shock generation
# ---------------------------------------------------------------------------

def _sobol_uniforms(
    n_samples: int,
    horizon: int,
    seed: int | None,
) -> np.ndarray:
    """
    Generate Sobol uniforms without triggering the power-of-2 warning.

    Sobol balance properties work best with random_base2.
    This function generates the next power of 2 and slices the required count.
    """

    if n_samples <= 0:
        return np.empty((0, horizon))

    m = int(np.ceil(np.log2(n_samples)))

    sobol_engine = qmc.Sobol(
        d=horizon,
        scramble=True,
        seed=seed,
    )

    u_full = sobol_engine.random_base2(m=m)
    u = u_full[:n_samples]

    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1.0 - eps)

    return u


def _normal_shocks(
    n_paths: int,
    horizon: int,
    seed: int | None,
) -> np.ndarray:
    """
    Generate standard-normal shocks using Sobol + antithetic variates.
    """

    half = n_paths // 2

    if half > 0:
        u = _sobol_uniforms(
            n_samples=half,
            horizon=horizon,
            seed=seed,
        )

        z_pos = scipy_norm.ppf(u)
        z_neg = -z_pos

        z = np.vstack([z_pos, z_neg])
    else:
        z = np.empty((0, horizon))

    if n_paths % 2 == 1:
        rng = np.random.default_rng(seed)
        extra = rng.standard_normal((1, horizon))
        z = np.vstack([z, extra])

    return z


def _student_t_shocks(
    log_returns: np.ndarray,
    n_paths: int,
    horizon: int,
    seed: int | None,
) -> tuple[np.ndarray, float]:
    """
    Generate Student-t shocks using Sobol + antithetic variates.

    The t-shocks are standardized so that the GBM sigma parameter remains
    interpretable as volatility.
    """

    mu = float(log_returns.mean())
    sigma = float(log_returns.std(ddof=1))

    if np.isnan(sigma) or sigma <= 0:
        raise ValueError("Cannot fit Student-t shocks because sigma is invalid.")

    standardized_returns = (log_returns - mu) / sigma

    try:
        df, loc, scale = scipy.stats.t.fit(
            standardized_returns,
            floc=0.0,
        )
    except Exception:
        df, loc, scale = 6.0, 0.0, 1.0

    if df <= 2:
        df = 2.1

    half = n_paths // 2

    if half > 0:
        u = _sobol_uniforms(
            n_samples=half,
            horizon=horizon,
            seed=seed,
        )

        z_pos = scipy.stats.t.ppf(
            u,
            df=df,
            loc=0.0,
            scale=scale,
        )

        theoretical_std = scale * np.sqrt(df / (df - 2.0))

        if theoretical_std > 0 and not np.isnan(theoretical_std):
            z_pos = z_pos / theoretical_std

        z_neg = -z_pos
        z = np.vstack([z_pos, z_neg])
    else:
        z = np.empty((0, horizon))

    if n_paths % 2 == 1:
        rng = np.random.default_rng(seed)

        extra = scipy.stats.t.rvs(
            df=df,
            loc=0.0,
            scale=scale,
            size=(1, horizon),
            random_state=rng,
        )

        theoretical_std = scale * np.sqrt(df / (df - 2.0))

        if theoretical_std > 0 and not np.isnan(theoretical_std):
            extra = extra / theoretical_std

        z = np.vstack([z, extra])

    return z, float(df)


# ---------------------------------------------------------------------------
# Optional regime override
# ---------------------------------------------------------------------------

def _try_apply_regime_override(
    series: pd.Series,
    mu: float,
    sigma: float,
) -> tuple[float, float, str | None, float | None]:
    """
    Optionally override mu/sigma using engines.regime.detect_regime().

    If hmmlearn is unavailable or regime detection fails, keep original mu/sigma.
    """

    try:
        regime_module = import_module("engines.regime")
        detect_regime = regime_module.detect_regime

        regime = detect_regime(series)

        if "unavailable" in regime.label.lower():
            return mu, sigma, regime.label, None

        regime_mu = float(regime.mu[regime.current_regime])
        regime_sigma = float(regime.sigma[regime.current_regime])
        regime_prob = float(regime.regime_probs[regime.current_regime])

        if np.isnan(regime_mu) or np.isnan(regime_sigma) or regime_sigma <= 0:
            return mu, sigma, "regime invalid; baseline used", None

        return regime_mu, regime_sigma, regime.label, regime_prob

    except Exception as exc:
        return mu, sigma, f"regime unavailable: {exc}", None


# ---------------------------------------------------------------------------
# Cached core implementation
# ---------------------------------------------------------------------------

def _forecast_impl(
    prices: np.ndarray,
    dates_str: list[str],
    symbol: str,
    horizon: int,
    n_paths: int,
    seed: int | None,
    frequency: str,
    calibration_window: int | None,
    distribution: str,
    use_regime: bool,
) -> dict:
    """
    Core GBM path simulation.

    Returns dictionary with:
        paths
        start_price
        mu
        sigma
        selected_calibration_window
        distribution
        student_t_df
        regime_label
        regime_prob
    """

    history = PriceHistory(
        dates=np.array(dates_str),
        prices=np.asarray(prices, dtype=float),
        symbol=symbol,
    )

    series = prepare_price_series(
        history=history,
        frequency=frequency,
    )

    if len(series) < 30:
        raise ValueError(
            f"Not enough observations after resampling to {frequency}. "
            f"Need at least 30, got {len(series)}."
        )

    if calibration_window is None:
        selected_window = select_calibration_window(series)
    else:
        selected_window = calibration_window

    if selected_window <= 1:
        raise ValueError("calibration_window must be greater than 1.")

    if len(series) < selected_window + 1:
        raise ValueError(
            f"Not enough prices for calibration_window={selected_window}. "
            f"Need {selected_window + 1}, got {len(series)}."
        )

    calibration_prices = series.iloc[-(selected_window + 1):]

    log_returns = np.diff(np.log(calibration_prices.values))

    mu = float(log_returns.mean())
    sigma = float(log_returns.std(ddof=1))

    if np.isnan(mu):
        raise ValueError("Invalid drift estimate: mu is NaN.")

    if np.isnan(sigma) or sigma <= 0:
        raise ValueError("Invalid volatility estimate: sigma must be positive.")

    regime_label = None
    regime_prob = None

    if use_regime:
        mu, sigma, regime_label, regime_prob = _try_apply_regime_override(
            series=calibration_prices,
            mu=mu,
            sigma=sigma,
        )

        if regime_prob is not None:
            print(f"[forecaster] Regime: {regime_label} (p={regime_prob:.2f})")
        else:
            print(f"[forecaster] Regime not applied: {regime_label}")

    distribution = distribution.lower()

    if distribution not in {"normal", "student-t"}:
        raise ValueError("distribution must be either 'normal' or 'student-t'.")

    student_t_df = None

    if distribution == "normal":
        z = _normal_shocks(
            n_paths=n_paths,
            horizon=horizon,
            seed=seed,
        )
    else:
        z, student_t_df = _student_t_shocks(
            log_returns=log_returns,
            n_paths=n_paths,
            horizon=horizon,
            seed=seed,
        )

        print(f"[forecaster] Student-t shocks enabled, df={student_t_df:.2f}")

    start_price = float(series.iloc[-1])

    increments = (mu - 0.5 * sigma**2) + sigma * z
    paths = start_price * np.exp(np.cumsum(increments, axis=1))

    return {
        "paths": paths,
        "start_price": start_price,
        "mu": mu,
        "sigma": sigma,
        "selected_calibration_window": selected_window,
        "distribution": distribution,
        "student_t_df": student_t_df,
        "regime_label": regime_label,
        "regime_prob": regime_prob,
    }


# ---------------------------------------------------------------------------
# Public forecast function
# ---------------------------------------------------------------------------

def forecast(
    history: PriceHistory,
    horizon: int,
    frequency: str = "M",
    n_paths: int = 8192,
    seed: int | None = 42,
    calibration_window: int | None = None,
    use_cache: bool = True,
    distribution: str = "normal",
    use_regime: bool = False,
    model: str = "xgb-garch-t",
) -> PriceForecast:
    """
    Unified forecaster entry point.

    Default:
        model="xgb-garch-t"

    Supported models:
        xgb-garch-t:
            XGBoost drift + GARCH(1,1)-t residual volatility.

        gbm / normal:
            Legacy GBM-Sobol-Antithetic normal shocks.

        student-t:
            Legacy GBM-Sobol-Antithetic Student-t shocks.

    Notes:
        GBM path is retained as a baseline/backtest fallback.
        Live Phase 3 pipeline should use XGB-GARCH-t.
    """

    model = model.lower().strip()
    frequency = frequency.upper()
    distribution = distribution.lower().strip()

    if horizon <= 0:
        raise ValueError("horizon must be greater than 0.")

    if n_paths <= 0:
        raise ValueError("n_paths must be greater than 0.")

    if model not in {"xgb-garch-t", "gbm", "normal", "student-t"}:
        raise ValueError(
            "model must be one of: 'xgb-garch-t', 'gbm', 'normal', 'student-t'."
        )

    # -----------------------------------------------------------------------
    # Phase 3 forecaster: XGB-GARCH-t
    # -----------------------------------------------------------------------

    if model == "xgb-garch-t":
        return forecast_xgb_garch_t(
            history=history,
            horizon=horizon,
            n_paths=n_paths,
            seed=seed,
            calibration_window=calibration_window or 1000,
            daily_steps=21,
            drift_scale=0.25,
            drift_clip=0.01,
            debug=False,
        )

    # -----------------------------------------------------------------------
    # Legacy GBM fallback / benchmark path
    # -----------------------------------------------------------------------

    if model == "student-t":
        distribution = "student-t"
    else:
        distribution = "normal"

    if frequency not in {"D", "W", "M"}:
        raise ValueError("frequency must be one of: 'D', 'W', 'M'.")

    if distribution not in {"normal", "student-t"}:
        raise ValueError("distribution must be either 'normal' or 'student-t'.")

    dates_str = list(pd.to_datetime(history.dates).strftime("%Y-%m-%d"))

    args = (
        np.asarray(history.prices, dtype=float),
        dates_str,
        history.symbol,
        horizon,
        n_paths,
        seed,
        frequency,
        calibration_window,
        distribution,
        use_regime,
    )

    if use_cache:
        cached_fn = _memory.cache(_forecast_impl)
        result = cached_fn(*args)
        print("[forecaster] Cache enabled.")
    else:
        result = _forecast_impl(*args)
        print("[forecaster] Cache disabled — computed fresh paths.")

    if use_regime and distribution == "student-t":
        model_name = "GBM-Sobol-Antithetic-HMM-t"
    elif use_regime:
        model_name = "GBM-Sobol-Antithetic-HMM"
    elif distribution == "student-t":
        model_name = "GBM-Sobol-Antithetic-t"
    else:
        model_name = "GBM-Sobol-Antithetic"

    return PriceForecast(
        paths=result["paths"],
        model_name=model_name,
        start_price=result["start_price"],
        mu=result["mu"],
        sigma=result["sigma"],
        frequency=frequency,
        seed=seed,
        calibration_window=result["selected_calibration_window"],
    )


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv("data/raw/wti_price_history.csv")

    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]
    df = df.sort_values("date")

    history = PriceHistory(
        dates=df["date"].to_numpy(),
        prices=df["price"].astype(float).to_numpy(),
        symbol="WTI",
    )

    result = forecast(
        history=history,
        horizon=6,
        frequency="M",
        n_paths=512,
        seed=42,
        calibration_window=1000,
        model="xgb-garch-t",
    )

    print("===== FORECAST OUTPUT =====")
    print(f"Model: {result.model_name}")
    print(f"Frequency: {result.frequency}")
    print(f"Start price: {result.start_price:.2f}")
    print(f"Drift / mu: {result.mu:.6f}")
    print(f"Volatility / sigma: {result.sigma:.6f}")
    print(f"Paths shape: {result.paths.shape}")
    print(f"Calibration window: {result.calibration_window}")

    print("\nFirst 5 simulated paths:")
    print(np.round(result.paths[:5], 2))

    print("\nFan chart quantiles:")
    quantiles = result.quantiles()

    for q, values in quantiles.items():
        print(f"P{q}: {np.round(values, 2)}")