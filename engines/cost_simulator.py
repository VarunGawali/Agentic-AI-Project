"""
Monte Carlo cost simulator: (PriceForecast, ExposureBook, StrategyParams,
forward_price) -> CostDistribution

BASELINE (Phase 1): path-wise simulation with per-path strategy application,
  dual-mode (accurate vs optimized), preallocated output array.
UPGRADE  (Phase 3): Sobol quasi-MC + antithetic variates.
"""

from __future__ import annotations
import numpy as np

from hedging_assistant.contracts import (
    PriceForecast, ExposureBook, StrategyParams, CostDistribution,
)
from hedging_assistant.engines.strategy_library import apply_strategy


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

    Args:
        forecast_obj: PriceForecast with paths (n_paths, horizon)
        exposure: ExposureBook with volumes (horizon,)
        params: StrategyParams defining the hedging rule
        forward_price: price locked in for hedged volumes (USD/bbl)
        cvar_alpha: tail level for CVaR computation (default 0.95 = worst 5%)
        mode: "accurate" applies strategy per-path (captures path-dependence);
              "optimized" applies strategy once on the first path (faster,
              equivalent for staggered which ignores path content)

    Returns:
        CostDistribution with per-path costs and summary statistics
    """
    # --- input validation ---
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

    # preallocate result array
    total_costs = np.empty(n_paths, dtype=float)

    if mode == "optimized":
        # staggered is path-independent — compute once, broadcast
        frac = apply_strategy(params, paths[0])
        if len(frac) != horizon:
            raise ValueError(
                f"apply_strategy returned {len(frac)} fractions; expected {horizon}"
            )
        if np.any(frac < 0) or np.any(frac > 1):
            raise ValueError("hedge fractions must be in [0, 1]")

        hedged_cost = float((frac * volumes * forward_price).sum())
        unhedged_costs = ((1.0 - frac) * volumes * paths).sum(axis=1)
        total_costs[:] = hedged_cost + unhedged_costs

    else:  # "accurate" — per-path strategy application
        for i in range(n_paths):
            frac = apply_strategy(params, paths[i])
            if len(frac) != horizon:
                raise ValueError(
                    f"apply_strategy returned {len(frac)} fractions; expected {horizon}"
                )
            if np.any(frac < 0) or np.any(frac > 1):
                raise ValueError(f"hedge fractions out of [0,1] on path {i}")

            hedged = (frac * volumes * forward_price).sum()
            unhedged = ((1.0 - frac) * volumes * paths[i]).sum()
            total_costs[i] = hedged + unhedged

    return CostDistribution(costs=total_costs, cvar_alpha=cvar_alpha)
