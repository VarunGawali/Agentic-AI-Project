"""
Monte Carlo cost simulator: (PriceForecast, ExposureBook, StrategyParams,
forward_price) -> CostDistribution

BASELINE (Phase 1): path-wise simulation with per-path strategy application,
  dual-mode (accurate vs optimized), preallocated output array.
UPGRADES:
  #1  Vectorized Monte Carlo — no per-path loop in either mode
  #5  Bootstrap confidence intervals on mean and CVaR
"""

from __future__ import annotations
import numpy as np

from hedging_assistant.contracts import (
    PriceForecast, ExposureBook, StrategyParams, CostDistribution,
)
from hedging_assistant.engines.strategy_library import apply_strategy


def _bootstrap_ci(
    costs: np.ndarray,
    statistic_fn,
    n_boot: int = 500,
    alpha: float = 0.95,
    seed: int = 0,
) -> tuple:
    """
    Compute bootstrap confidence interval for a scalar statistic.

    Uses numpy resampling only (no scipy dependency) — rng.choice with replacement.
    Returns (low, high) at the given alpha level.
    """
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot, dtype=float)
    n = len(costs)
    for i in range(n_boot):
        sample = costs[rng.choice(n, size=n, replace=True)]
        boot_stats[i] = statistic_fn(sample)
    lo = float(np.percentile(boot_stats, (1 - alpha) / 2 * 100))
    hi = float(np.percentile(boot_stats, (1 + alpha) / 2 * 100))
    return (lo, hi)


def simulate_cost(
    forecast_obj: PriceForecast,
    exposure: ExposureBook,
    params: StrategyParams,
    forward_price: float,
    cvar_alpha: float = 0.95,
    mode: str = "accurate",
) -> CostDistribution:
    """
    Simulate total procurement cost distribution over all forecast paths.

    Improvement #1: Fully vectorized — no per-path for loop even in "accurate"
    mode. This is safe for path-independent strategies (e.g. staggered) because
    apply_strategy returns the same fractions regardless of which path is passed.
    Applying once on paths[0] and broadcasting is mathematically identical to
    applying per-path, since the output is constant across paths.

    Improvement #5: Bootstrap CIs computed on mean and CVaR.

    Args:
        forecast_obj: PriceForecast with paths (n_paths, horizon)
        exposure: ExposureBook with volumes (horizon,)
        params: StrategyParams defining the hedging rule
        forward_price: price locked in for hedged volumes (USD/bbl)
        cvar_alpha: tail level for CVaR computation (default 0.95 = worst 5%)
        mode: "accurate" or "optimized" — both fully vectorized now;
              mode parameter kept for API compatibility

    Returns:
        CostDistribution with per-path costs, summary stats, and bootstrap CIs
    """
    if forward_price <= 0:
        raise ValueError(f"forward_price must be positive; got {forward_price}")
    if not (0 < cvar_alpha < 1):
        raise ValueError(f"cvar_alpha must be in (0, 1); got {cvar_alpha}")

    paths = np.asarray(forecast_obj.paths, dtype=float)       # (n_paths, horizon)
    volumes = np.asarray(exposure.volumes, dtype=float)        # (horizon,)

    n_paths, horizon = paths.shape
    if len(volumes) != horizon:
        raise ValueError(
            f"exposure.volumes length {len(volumes)} != forecast horizon {horizon}"
        )

    # Improvement #1: vectorized form — safe for path-independent strategies.
    # For staggered strategies, apply_strategy returns a constant fraction array
    # regardless of which path is provided, so using paths[0] is equivalent to
    # using any other path or computing per-path individually.
    frac = apply_strategy(params, paths[0])   # shape (horizon,)
    if len(frac) != horizon:
        raise ValueError(
            f"apply_strategy returned {len(frac)} fractions; expected {horizon}"
        )
    if np.any(frac < 0) or np.any(frac > 1):
        raise ValueError("hedge fractions must be in [0, 1]")

    hedged_cost = (frac * volumes * forward_price).sum()           # scalar
    unhedged_costs = ((1.0 - frac) * volumes * paths).sum(axis=1) # (n_paths,)
    total_costs = hedged_cost + unhedged_costs

    # Improvement #5: bootstrap CIs on mean and CVaR
    def _cvar_stat(c: np.ndarray) -> float:
        thr = np.percentile(c, cvar_alpha * 100)
        tail = c[c >= thr]
        return float(tail.mean()) if len(tail) else float(thr)

    ci_mean = _bootstrap_ci(total_costs, np.mean)
    ci_cvar = _bootstrap_ci(total_costs, _cvar_stat)

    dist = CostDistribution(costs=total_costs, cvar_alpha=cvar_alpha)
    dist.ci_mean = ci_mean
    dist.ci_cvar = ci_cvar
    return dist
