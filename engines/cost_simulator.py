"""
Monte Carlo cost simulator.

Input:
    PriceForecast + ExposureBook + StrategyParams + forward price/curve

Output:
    CostDistribution

Supports:
    - Scalar forward price
    - ForwardCurve / forward-price array
    - Vectorized simulation for path-independent strategies
    - Path-wise simulation for path-dependent strategies
    - Bootstrap confidence intervals
    - Marginal CVaR decomposition by period

Phase 3:
    - Uses strategy_library.is_path_dependent()
    - STAGGERED remains fully vectorized
    - TRIGGER / VOLATILITY / HYBRID / DP_OPTIMAL run path-wise
"""

from __future__ import annotations

import numpy as np

from hedging_assistant.contracts import (
    PriceForecast,
    ExposureBook,
    StrategyParams,
    CostDistribution,
)

from hedging_assistant.engines.strategy_library import apply_strategy, is_path_dependent
from hedging_assistant.engines.utils import resolve_forward_curve


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    costs: np.ndarray,
    statistic_fn,
    n_boot: int = 500,
    alpha: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for a scalar statistic.
    """

    costs = np.asarray(costs, dtype=float)

    if costs.ndim != 1:
        raise ValueError("costs must be 1D for bootstrap")

    if len(costs) == 0:
        raise ValueError("costs cannot be empty for bootstrap")

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    rng = np.random.default_rng(seed)
    n = len(costs)
    # Draw all bootstrap indices at once — one vectorised call instead of n_boot loops.
    all_idx = rng.integers(0, n, size=(n_boot, n))
    boot_stats = np.array([statistic_fn(costs[idx]) for idx in all_idx], dtype=float)

    lo = float(np.percentile(boot_stats, (1 - alpha) / 2 * 100))
    hi = float(np.percentile(boot_stats, (1 + alpha) / 2 * 100))

    return lo, hi


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    paths: np.ndarray,
    volumes: np.ndarray,
    exposure: ExposureBook,
    cvar_alpha: float,
) -> tuple[int, int]:
    """
    Shared validation for cost simulation.
    """

    if paths.ndim != 2:
        raise ValueError("forecast_obj.paths must be 2D: (n_paths, horizon)")

    n_paths, horizon = paths.shape

    if n_paths == 0:
        raise ValueError("forecast_obj.paths cannot have zero paths")

    if horizon == 0:
        raise ValueError("forecast horizon cannot be zero")

    if volumes.ndim != 1:
        raise ValueError("exposure.volumes must be 1D")

    if len(volumes) != horizon:
        raise ValueError(
            f"Exposure horizon mismatch: exposure has {len(volumes)} periods, "
            f"but forecast has {horizon} periods"
        )

    if hasattr(exposure, "horizon") and exposure.horizon != horizon:
        raise ValueError(
            f"ExposureBook.horizon mismatch: exposure.horizon={exposure.horizon}, "
            f"forecast horizon={horizon}"
        )

    if np.any(paths <= 0):
        raise ValueError("forecast paths must contain only positive prices")

    if np.any(volumes < 0):
        raise ValueError("exposure volumes cannot be negative")

    if not 0 < cvar_alpha < 1:
        raise ValueError(f"cvar_alpha must be between 0 and 1; got {cvar_alpha}")

    return n_paths, horizon


def _validate_hedge_fractions(
    hedge_fractions: np.ndarray,
    horizon: int,
) -> None:
    """
    Validate strategy output.
    """

    hedge_fractions = np.asarray(hedge_fractions, dtype=float)

    if hedge_fractions.ndim != 1:
        raise ValueError("hedge fractions must be a 1D array")

    if len(hedge_fractions) != horizon:
        raise ValueError(
            f"Strategy returned {len(hedge_fractions)} hedge fractions, "
            f"expected {horizon}"
        )

    if np.any((hedge_fractions < 0) | (hedge_fractions > 1)):
        raise ValueError("hedge fractions must be between 0 and 1")


def _cvar_stat(
    costs: np.ndarray,
    cvar_alpha: float,
) -> float:
    """
    CVaR = average of worst upper-tail costs.
    """

    costs = np.asarray(costs, dtype=float)

    threshold = np.percentile(costs, cvar_alpha * 100)
    tail = costs[costs >= threshold]

    return float(tail.mean()) if len(tail) else float(threshold)


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

def simulate_cost(
    forecast_obj: PriceForecast,
    exposure: ExposureBook,
    params: StrategyParams,
    forward_price,
    cvar_alpha: float = 0.95,
    mode: str = "optimized",
    compute_ci: bool = False,
    n_boot: int = 500,
    ci_alpha: float = 0.95,
    seed: int = 0,
    price_history: "np.ndarray | None" = None,
    long_run_vol: "float | None" = None,
) -> CostDistribution:
    """
    Simulate total procurement cost distribution.

    Formula:
        total_cost =
            sum_t(hedge_fraction_t * volume_t * forward_price_t)
            +
            sum_t((1 - hedge_fraction_t) * volume_t * spot_price_t)

    Phase 3 behavior:
        If strategy is path-dependent, simulator automatically uses path-wise
        accurate evaluation even if mode='optimized' is passed.
    """

    mode = mode.lower()

    if mode not in {"optimized", "accurate"}:
        raise ValueError("mode must be either 'optimized' or 'accurate'")

    paths = np.asarray(forecast_obj.paths, dtype=float)
    volumes = np.asarray(exposure.volumes, dtype=float)

    _, horizon = _validate_inputs(
        paths=paths,
        volumes=volumes,
        exposure=exposure,
        cvar_alpha=cvar_alpha,
    )

    fwd_curve = resolve_forward_curve(
        forward_price=forward_price,
        horizon=horizon,
    )

    if is_path_dependent(params):
        total_costs = _simulate_cost_accurate(
            paths=paths,
            volumes=volumes,
            params=params,
            fwd_curve=fwd_curve,
            price_history=price_history,
            long_run_vol=long_run_vol,
        )

    else:
        total_costs = _simulate_cost_optimized(
            paths=paths,
            volumes=volumes,
            params=params,
            fwd_curve=fwd_curve,
        )

    dist = CostDistribution(
        costs=total_costs,
        cvar_alpha=cvar_alpha,
    )

    if compute_ci:
        dist.ci_mean = _bootstrap_ci(
            costs=total_costs,
            statistic_fn=np.mean,
            n_boot=n_boot,
            alpha=ci_alpha,
            seed=seed,
        )

        dist.ci_cvar = _bootstrap_ci(
            costs=total_costs,
            statistic_fn=lambda c: _cvar_stat(c, cvar_alpha),
            n_boot=n_boot,
            alpha=ci_alpha,
            seed=seed,
        )

    return dist


# ---------------------------------------------------------------------------
# Optimized vectorized simulation
# ---------------------------------------------------------------------------

def _simulate_cost_optimized(
    paths: np.ndarray,
    volumes: np.ndarray,
    params: StrategyParams,
    fwd_curve: np.ndarray,
) -> np.ndarray:
    """
    Vectorized simulation.

    Safe for path-independent strategies like STAGGERED.
    """

    horizon = paths.shape[1]

    hedge_fractions = apply_strategy(
        params=params,
        price_path=paths[0],
    )

    _validate_hedge_fractions(
        hedge_fractions=hedge_fractions,
        horizon=horizon,
    )

    hedged_cost = np.sum(
        hedge_fractions * volumes * fwd_curve
    )

    unhedged_costs = np.sum(
        (1.0 - hedge_fractions) * volumes * paths,
        axis=1,
    )

    return hedged_cost + unhedged_costs


# ---------------------------------------------------------------------------
# Accurate path-wise simulation
# ---------------------------------------------------------------------------

def _simulate_cost_accurate(
    paths: np.ndarray,
    volumes: np.ndarray,
    params: StrategyParams,
    fwd_curve: np.ndarray,
    price_history: "np.ndarray | None" = None,
    long_run_vol: "float | None" = None,
) -> np.ndarray:
    """
    Path-wise simulation.

    Required for:
        TRIGGER
        VOLATILITY
        HYBRID
        DP_OPTIMAL
    """

    n_paths, horizon = paths.shape
    total_costs = np.empty(n_paths, dtype=float)

    for i in range(n_paths):
        price_path = paths[i]

        hedge_fractions = apply_strategy(
            params=params,
            price_path=price_path,
            price_history=price_history,
            long_run_vol=long_run_vol,
        )

        _validate_hedge_fractions(
            hedge_fractions=hedge_fractions,
            horizon=horizon,
        )

        hedged_cost = hedge_fractions * volumes * fwd_curve

        unhedged_cost = (
            (1.0 - hedge_fractions)
            * volumes
            * price_path
        )

        total_costs[i] = np.sum(hedged_cost + unhedged_cost)

    return total_costs


# ---------------------------------------------------------------------------
# Marginal CVaR by period
# ---------------------------------------------------------------------------

def marginal_cvar(
    forecast_obj: PriceForecast,
    exposure: ExposureBook,
    params: StrategyParams,
    forward_price,
    cvar_alpha: float = 0.95,
    mode: str = "optimized",
) -> np.ndarray:
    """
    Marginal CVaR decomposition by period.

    Returns:
        np.ndarray of shape (horizon,)

    Interpretation:
        Each value shows average period-level cost contribution inside the
        worst CVaR tail scenarios.
    """

    mode = mode.lower()

    if mode not in {"optimized", "accurate"}:
        raise ValueError("mode must be either 'optimized' or 'accurate'")

    paths = np.asarray(forecast_obj.paths, dtype=float)
    volumes = np.asarray(exposure.volumes, dtype=float)

    _, horizon = _validate_inputs(
        paths=paths,
        volumes=volumes,
        exposure=exposure,
        cvar_alpha=cvar_alpha,
    )

    fwd_curve = resolve_forward_curve(
        forward_price=forward_price,
        horizon=horizon,
    )

    if is_path_dependent(params):
        period_costs = _period_costs_accurate(
            paths=paths,
            volumes=volumes,
            params=params,
            fwd_curve=fwd_curve,
        )
    else:
        period_costs = _period_costs_optimized(
            paths=paths,
            volumes=volumes,
            params=params,
            fwd_curve=fwd_curve,
        )

    total_costs = period_costs.sum(axis=1)

    threshold = np.percentile(total_costs, cvar_alpha * 100)
    tail_mask = total_costs >= threshold

    if not np.any(tail_mask):
        return np.zeros(horizon, dtype=float)

    tail_period_costs = period_costs[tail_mask]

    return tail_period_costs.mean(axis=0)


def _period_costs_optimized(
    paths: np.ndarray,
    volumes: np.ndarray,
    params: StrategyParams,
    fwd_curve: np.ndarray,
) -> np.ndarray:
    """
    Vectorized per-period costs for path-independent strategies.

    Returns:
        array of shape (n_paths, horizon)
    """

    horizon = paths.shape[1]

    hedge_fractions = apply_strategy(
        params=params,
        price_path=paths[0],
    )

    _validate_hedge_fractions(
        hedge_fractions=hedge_fractions,
        horizon=horizon,
    )

    period_costs = (
        hedge_fractions * volumes * fwd_curve
        + (1.0 - hedge_fractions) * volumes * paths
    )

    return period_costs


def _period_costs_accurate(
    paths: np.ndarray,
    volumes: np.ndarray,
    params: StrategyParams,
    fwd_curve: np.ndarray,
) -> np.ndarray:
    """
    Path-wise per-period costs for path-dependent strategies.

    Returns:
        array of shape (n_paths, horizon)
    """

    n_paths, horizon = paths.shape
    period_costs = np.empty((n_paths, horizon), dtype=float)

    for i in range(n_paths):
        price_path = paths[i]

        hedge_fractions = apply_strategy(
            params=params,
            price_path=price_path,
        )

        _validate_hedge_fractions(
            hedge_fractions=hedge_fractions,
            horizon=horizon,
        )

        period_costs[i] = (
            hedge_fractions * volumes * fwd_curve
            + (1.0 - hedge_fractions) * volumes * price_path
        )

    return period_costs


# ---------------------------------------------------------------------------
# Backward-compatible wrapper
# ---------------------------------------------------------------------------

def simulate_cost_fast(
    forecast_obj: PriceForecast,
    exposure: ExposureBook,
    params: StrategyParams,
    forward_price,
    cvar_alpha: float = 0.95,
) -> CostDistribution:
    """
    Backward-compatible wrapper.

    Old test scripts can still call simulate_cost_fast().
    """

    return simulate_cost(
        forecast_obj=forecast_obj,
        exposure=exposure,
        params=params,
        forward_price=forward_price,
        cvar_alpha=cvar_alpha,
        mode="optimized",
        compute_ci=False,
    )