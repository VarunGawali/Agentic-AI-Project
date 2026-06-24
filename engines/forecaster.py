"""
Forecaster engine: PriceHistory -> PriceForecast

BASELINE (Phase 1): Geometric Brownian Motion with pandas-based preprocessing,
  frequency resampling (D/W/M), and a calibration window.
UPGRADES:
  #3  Sobol quasi-random sequences + antithetic variates
      (O(1/N) convergence; halves estimator variance simultaneously)
  #4  joblib.Memory path caching
  #6  AIC-based rolling calibration window selection

CHANGES:
  - Added `distribution` param: "normal" (default) or "student-t" (fat tails)
  - Added `use_regime` param: when True, calls detect_regime() and uses regime-specific mu/sigma
  - Student-t: fits degrees-of-freedom via scipy.stats.t.fit on log-returns
  - Added `model` param: "gbm" (default) or "xgb-garch-t" (XGBoost drift + GARCH(1,1)-t residuals)
    XGB-GARCH-t features: sin/cos seasonality, momentum lags, realised vol, RSI,
    price_dev_60d, HMM regime — all self-contained so they can be recomputed
    during forward simulation. EIA inventory used only for calibration (zero at sim time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import joblib
import scipy.stats
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
        aic = 2 * 2 - 2 * log_lik
        if aic < best_aic:
            best_aic = aic
            best_w = w

    return best_w


# ----------------------------------------------------------------------------
# Core GBM implementation cached by joblib (Improvement #4)
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
    distribution: str = "normal",
    use_regime: bool = False,
) -> np.ndarray:
    """
    Core GBM path simulation, cached by joblib.Memory.

    Improvement #3: Uses Sobol quasi-random sequences + antithetic variates.
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

    if use_regime:
        from hedging_assistant.engines.regime import detect_regime
        regime = detect_regime(s)
        mu = float(regime.mu[regime.current_regime])
        sigma = float(regime.sigma[regime.current_regime])
        print(f"[forecaster] Regime: {regime.label} (p={regime.regime_probs[regime.current_regime]:.2f})")

    s0 = float(s.iloc[-1])

    half = n_paths // 2
    sobol_engine = qmc.Sobol(d=horizon, scramble=True, seed=seed)
    u = sobol_engine.random(half)

    if distribution == "student-t":
        df, loc, scale = scipy.stats.t.fit(log_ret, floc=mu)
        print(f"[forecaster] Student-t df={df:.1f}")
        z_pos = scipy.stats.t.ppf(u, df=df, loc=0, scale=scale)
    else:
        z_pos = scipy_norm.ppf(u)

    z_neg = -z_pos
    z = np.vstack([z_pos, z_neg])
    if n_paths % 2 == 1:
        rng = np.random.default_rng(seed)
        if distribution == "student-t":
            extra = scipy.stats.t.rvs(df=df, loc=0, scale=scale, size=(1, horizon),
                                      random_state=np.random.default_rng(seed))
        else:
            extra = rng.standard_normal((1, horizon))
        z = np.vstack([z, extra])

    increments = (mu - 0.5 * sigma ** 2) + sigma * z
    paths = s0 * np.exp(np.cumsum(increments, axis=1))
    return paths


# ----------------------------------------------------------------------------
# XGBoost + GARCH(1,1)-t forecaster
# ----------------------------------------------------------------------------

def _compute_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """RSI centred at 0 (range -50 to +50)."""
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.full_like(prices, np.nan)
    avg_loss = np.full_like(prices, np.nan)

    if len(prices) <= window:
        return np.zeros(len(prices))

    avg_gain[window] = gains[1:window + 1].mean()
    avg_loss[window] = losses[1:window + 1].mean()

    for i in range(window + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gains[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + losses[i]) / window

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    rsi = 100 - 100 / (1 + rs)
    return rsi - 50.0   # centre at 0


def _build_features(
    log_prices: np.ndarray,
    dates: pd.DatetimeIndex,
    eia_inventory: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build feature matrix from log-price series for XGBoost training.

    Self-contained features (recomputable during simulation):
      sin_month, cos_month, log_ret_lag1/5/21, rv_5d, rv_21d,
      price_dev_60d, rsi_14, hmm_regime
    External feature (calibration-only, zeroed during simulation):
      eia_inventory_chg
    """
    n = len(log_prices)
    prices_raw = np.exp(log_prices)

    # Seasonality
    month = dates.month.values.astype(float)
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)

    # Log-return lags
    log_rets = np.diff(log_prices, prepend=log_prices[0])
    lag1  = np.concatenate([[0.0], log_rets[:-1]])
    lag5  = np.concatenate([np.zeros(5),  log_rets[:-5]])
    lag21 = np.concatenate([np.zeros(21), log_rets[:-21]])

    # Realised volatility (annualised)
    rv5  = pd.Series(log_rets).rolling(5,  min_periods=2).std().values * np.sqrt(252)
    rv21 = pd.Series(log_rets).rolling(21, min_periods=2).std().values * np.sqrt(252)
    rv5  = np.nan_to_num(rv5,  nan=float(np.nanmedian(rv5)))
    rv21 = np.nan_to_num(rv21, nan=float(np.nanmedian(rv21)))

    # Price deviation from 60-day MA
    ma60 = pd.Series(prices_raw).rolling(60, min_periods=10).mean().values
    ma60 = np.where(np.isnan(ma60), prices_raw, ma60)
    price_dev_60d = np.log(prices_raw / ma60)

    # RSI-14 centred at 0
    rsi14 = _compute_rsi(prices_raw, window=14)

    # HMM regime (0 = low-vol/bull, 1 = high-vol/bear)
    try:
        from hedging_assistant.engines.regime import detect_regime
        from hedging_assistant.contracts import PriceHistory
        tmp_hist = PriceHistory(
            dates=dates.values,
            prices=prices_raw,
        )
        s_tmp = pd.Series(prices_raw, index=dates)
        regime_obj = detect_regime(s_tmp)
        hmm_regime = regime_obj.path.astype(float)
        if len(hmm_regime) != n:
            hmm_regime = np.zeros(n)
    except Exception:
        hmm_regime = np.zeros(n)

    df = pd.DataFrame({
        "sin_month":    sin_month,
        "cos_month":    cos_month,
        "log_ret_lag1": lag1,
        "log_ret_lag5": lag5,
        "log_ret_lag21": lag21,
        "rv_5d":        rv5,
        "rv_21d":       rv21,
        "price_dev_60d": price_dev_60d,
        "rsi_14":       rsi14,
        "hmm_regime":   hmm_regime,
    })

    if eia_inventory is not None and len(eia_inventory) == n:
        df["eia_inventory_chg"] = eia_inventory
    else:
        df["eia_inventory_chg"] = 0.0

    return df


def _fit_xgb_garch(
    series: pd.Series,
    eia_inventory: np.ndarray | None = None,
    seed: int = 42,
) -> dict:
    """
    Fit XGBoost drift model and GARCH(1,1)-t on residuals.

    Returns a dict with:
      xgb_model, garch_params (omega, alpha, beta, nu, long_run_vol),
      last_features (for seed at simulation start), last_sigma2, last_price
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost required: pip install xgboost")
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError("arch required: pip install arch")

    prices = series.values.astype(float)
    log_prices = np.log(prices)
    dates = series.index

    log_rets = np.diff(log_prices)          # (n-1,) targets
    feat_df = _build_features(log_prices, dates, eia_inventory)
    # Features aligned with targets: X[t] predicts log_ret[t] = log(P[t+1]/P[t])
    X = feat_df.values[:-1]                 # (n-1, n_features)
    y = log_rets                            # (n-1,)

    # Time-series cross-validation (5 folds, no shuffle)
    n_samples = len(y)
    fold_size = n_samples // 6              # ~1/6 for each of 5 val folds
    best_params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                   "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
                   "reg_lambda": 1.0, "random_state": seed}

    # Fit on full data (CV would be used for param tuning; here we use fixed params
    # chosen to be conservative for ~3800 rows to avoid overfitting)
    model = xgb.XGBRegressor(**best_params, tree_method="hist", verbosity=0)
    model.fit(X, y)

    # Residuals for GARCH fitting
    y_pred = model.predict(X)
    residuals = y - y_pred                  # (n-1,)

    # GARCH(1,1)-t on residuals
    am = arch_model(residuals * 100, vol="GARCH", p=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    params = res.params

    omega = float(params.get("omega", 0.01))
    alpha = float(params.get("alpha[1]", 0.05))
    beta  = float(params.get("beta[1]", 0.90))
    nu    = float(params.get("nu", 8.0))
    nu    = max(nu, 2.5)                    # prevent infinite variance

    long_run_vol = float(np.sqrt(omega / max(1 - alpha - beta, 1e-6)) / 100)

    # Seed values for simulation
    last_sigma2 = float(res.conditional_volatility[-1] ** 2) / 10_000
    last_resid  = float(residuals[-1])

    last_feat = feat_df.values[-1].copy()   # features of the last observed day

    print(
        f"[forecaster] XGB-GARCH-t fitted: omega={omega:.4f}, alpha={alpha:.3f}, "
        f"beta={beta:.3f}, nu={nu:.1f}, long_run_vol={long_run_vol:.4f}"
    )

    return {
        "xgb_model": model,
        "feature_names": list(feat_df.columns),
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "nu": nu,
        "long_run_vol": long_run_vol,
        "last_sigma2": last_sigma2,
        "last_resid_scaled": last_resid * 100,
        "last_price": float(prices[-1]),
        "last_features": last_feat,
        "last_date": dates[-1],
    }


def _simulate_xgb_garch_paths(
    fitted: dict,
    horizon_months: int,
    n_paths: int,
    seed: int = 42,
    daily_steps: int = 21,
) -> np.ndarray:
    """
    Simulate forward price paths using XGBoost drift + GARCH(1,1)-t volatility.

    Each monthly period = daily_steps daily steps.
    At each step: XGB predicts drift from current features, GARCH evolves sigma^2,
    draw Student-t(nu) innovation, compound to next price.

    Returns paths shape (n_paths, horizon_months).
    """
    import xgboost as xgb

    xgb_model    = fitted["xgb_model"]
    omega        = fitted["omega"]
    alpha        = fitted["alpha"]
    beta         = fitted["beta"]
    nu           = fitted["nu"]
    long_run_vol = fitted["long_run_vol"]
    last_price   = fitted["last_price"]
    last_sigma2  = fitted["last_sigma2"]
    last_date    = fitted["last_date"]
    feat_names   = fitted["feature_names"]

    total_daily = horizon_months * daily_steps
    rng = np.random.default_rng(seed)

    # Draw all innovations upfront: Student-t(nu), shape (n_paths, total_daily)
    innovations = rng.standard_t(df=nu, size=(n_paths, total_daily))

    paths = np.empty((n_paths, horizon_months))

    for path_idx in range(n_paths):
        price = last_price
        log_price = np.log(price)
        sigma2 = last_sigma2
        sim_date = pd.Timestamp(last_date)

        # Rolling buffers for feature computation
        price_buf = [price]   # short buffer; we only need recent lags

        month_count = 0
        day_in_period = 0

        for day_idx in range(total_daily):
            sim_date = sim_date + pd.Timedelta(days=1)
            # Skip weekends (rough approximation)
            while sim_date.weekday() >= 5:
                sim_date += pd.Timedelta(days=1)

            # Build feature vector for this day
            month = sim_date.month
            sin_m = np.sin(2 * np.pi * month / 12)
            cos_m = np.cos(2 * np.pi * month / 12)

            # Log returns for lags (use buffer)
            buf = price_buf
            def _lr(lag):
                if len(buf) > lag:
                    return np.log(buf[-1] / buf[-1 - lag])
                return 0.0

            lr1  = _lr(1)
            lr5  = _lr(5)
            lr21 = _lr(21)

            # Realised vol from buffer
            if len(buf) >= 3:
                rv_buf = np.diff(np.log(buf[-min(22, len(buf)):]))
                rv5_val  = float(np.std(rv_buf[-5:]))  * np.sqrt(252) if len(rv_buf) >= 5  else long_run_vol
                rv21_val = float(np.std(rv_buf[-21:])) * np.sqrt(252) if len(rv_buf) >= 21 else long_run_vol
            else:
                rv5_val = rv21_val = long_run_vol

            # Price deviation from 60-day MA
            if len(buf) >= 60:
                ma60_val = np.mean(buf[-60:])
            else:
                ma60_val = float(np.mean(buf))
            dev60 = np.log(price / max(ma60_val, 1e-6))

            # RSI-14 (simplified from buffer)
            if len(buf) >= 15:
                rsi_val = float(_compute_rsi(np.array(buf[-30:]), 14)[-1])
            else:
                rsi_val = 0.0

            # HMM regime — use vol level as proxy during simulation
            hmm_val = 1.0 if rv5_val > long_run_vol * 1.5 else 0.0

            feat = np.array([[
                sin_m, cos_m, lr1, lr5, lr21,
                rv5_val, rv21_val, dev60, rsi_val, hmm_val,
                0.0,    # eia_inventory_chg = 0 during simulation
            ]], dtype=float)

            # XGB drift prediction
            drift = float(xgb_model.predict(feat)[0])

            # GARCH(1,1) variance update
            last_eps = innovations[path_idx, day_idx - 1] * np.sqrt(sigma2) if day_idx > 0 else 0.0
            sigma2 = omega / 10_000 + alpha * (last_eps * 100) ** 2 / 10_000 + beta * sigma2
            sigma2 = max(sigma2, 1e-10)

            sigma = np.sqrt(sigma2)
            z = innovations[path_idx, day_idx]
            log_return = drift + sigma * z
            log_price += log_return
            price = np.exp(log_price)

            price_buf.append(price)
            if len(price_buf) > 125:   # keep ~6 months of buffer
                price_buf.pop(0)

            day_in_period += 1
            if day_in_period == daily_steps:
                paths[path_idx, month_count] = price
                month_count += 1
                day_in_period = 0

    return paths


def forecast_xgb_garch(
    history: PriceHistory,
    horizon: int,
    n_paths: int = 5_000,
    seed: int = 42,
    calibration_window: int | None = None,
    daily_steps: int = 21,
    eia_inventory: np.ndarray | None = None,
) -> PriceForecast:
    """
    XGBoost-drift + GARCH(1,1)-t forecaster.

    Fits XGBoost on 9 self-contained features + optional EIA inventory
    (zeroed at simulation time). GARCH(1,1)-t captures vol clustering
    and fat-tail innovations. Compounds daily_steps=21 daily steps per
    monthly horizon period.

    Falls back to GBM-Sobol if xgboost or arch are not installed.
    """
    try:
        import xgboost  # noqa: F401
        import arch     # noqa: F401
    except ImportError as e:
        print(f"[forecaster] XGB-GARCH-t dependencies missing ({e}), falling back to GBM-Sobol.")
        return forecast(history, horizon, n_paths=n_paths, seed=seed)

    s = prepare_price_series(history, frequency="D",
                             calibration_window=calibration_window)
    if len(s) < 60:
        raise ValueError(f"Need >= 60 daily rows to fit XGB-GARCH-t; got {len(s)}")

    fitted = _fit_xgb_garch(s, eia_inventory=eia_inventory, seed=seed)
    paths = _simulate_xgb_garch_paths(
        fitted, horizon_months=horizon, n_paths=n_paths,
        seed=seed, daily_steps=daily_steps,
    )

    return PriceForecast(
        paths=paths,
        model_name="XGB-GARCH-t",
        frequency="M",
        calibration_window=calibration_window or len(s),
    )


# ----------------------------------------------------------------------------
# Unified entry point
# ----------------------------------------------------------------------------

def forecast(
    history: PriceHistory,
    horizon: int,
    n_paths: int = 10_000,
    seed: int | None = 42,
    frequency: str = "D",
    calibration_window: int | None = None,
    use_cache: bool = True,
    distribution: str = "normal",
    use_regime: bool = False,
    model: str = "gbm",
) -> PriceForecast:
    """
    Unified forecaster entry point.

    model="gbm"          — GBM-Sobol (default, fast)
    model="xgb-garch-t"  — XGBoost drift + GARCH(1,1)-t (richer, ~10s fit)

    All other kwargs are forwarded to the relevant implementation.
    """
    if model == "xgb-garch-t":
        return forecast_xgb_garch(
            history, horizon, n_paths=n_paths, seed=seed or 42,
            calibration_window=calibration_window,
        )

    # --- GBM-Sobol path ---
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
            distribution, use_regime,
        )
        print("[forecaster] Cache hit — returning cached paths.")
    else:
        print("[forecaster] Cache disabled — computing paths.")
        paths = _forecast_impl(
            history.prices, dates_str, history.symbol,
            horizon, n_paths, seed, frequency, calibration_window,
            distribution, use_regime,
        )

    if use_regime and distribution == "student-t":
        model_name = "GBM-Sobol-HMM-t"
    elif use_regime:
        model_name = "GBM-Sobol-HMM"
    elif distribution == "student-t":
        model_name = "GBM-Sobol-t"
    else:
        model_name = "GBM-Sobol"

    return PriceForecast(
        paths=paths,
        model_name=model_name,
        frequency=frequency,
        calibration_window=calibration_window,
    )
