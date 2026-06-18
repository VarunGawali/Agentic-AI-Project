"""
Scorer: evaluates and ranks hedging candidates.

BASELINE (Phase 1): sweeps a hedge-ratio grid, simulates each, scores on
  cost + CVaR (2-factor). Opportunity cost and execution risk are stubbed at 0.
UPGRADE:
  #2  Multiprocessing candidate sweep via ProcessPoolExecutor
"""

from __future__ import annotations
import os
import numpy as np
import concurrent.futures

from hedging_assistant.contracts import (
    PriceForecast, ExposureBook, RiskAppetite,
    StrategyType, StrategyParams, CostDistribution, FactorScore,
)
from hedging_assistant.engines.cost_simulator import simulate_cost

DEFAULT_HEDGE_GRID = np.linspace(0.0, 1.0, 11)   # 0%, 10%, ..., 100%


def score_policy(
    cost: CostDistribution,
    no_hedge_cost: CostDistribution,
    params: StrategyParams,
    risk: RiskAppetite,
) -> FactorScore:
    """
    Score a single policy against the no-hedge baseline.

    BASELINE: 2-factor (cost + lambda*CVaR). Opportunity cost and execution risk
    are set to 0 until Phase 3.
    """
    cost_factor = cost.mean
    cvar_factor = cost.cvar
    opp = 0.0    # Phase 3: regret when prices fall and we over-hedged
    exe = 0.0    # Phase 3: operational execution difficulty proxy
    blended = risk.w_cost * cost_factor + risk.w_cvar * cvar_factor
    return FactorScore(
        cost=cost_factor,
        cvar=cvar_factor,
        opportunity_cost=opp,
        execution_risk=exe,
        blended=blended,
    )


# Module-level worker function required for pickling with ProcessPoolExecutor
def _evaluate_single_candidate(args):
    """Worker for parallel candidate evaluation. Must be module-level for pickling."""
    ratio, forecast_paths, exposure_volumes, forward_price, cvar_alpha, max_hedge = args

    import numpy as np
    from hedging_assistant.contracts import (
        PriceForecast, ExposureBook, StrategyType, StrategyParams,
    )
    from hedging_assistant.engines.cost_simulator import simulate_cost

    fc_obj = PriceForecast(paths=forecast_paths, model_name="GBM-Sobol")
    exp_obj = ExposureBook(volumes=exposure_volumes)
    params = StrategyParams(
        strategy_type=StrategyType.STAGGERED,
        base_fraction=float(ratio),
        cap=float(max_hedge),
    )
    cost = simulate_cost(fc_obj, exp_obj, params, forward_price, cvar_alpha=cvar_alpha)
    return ratio, params, cost


def evaluate_candidates(
    forecast_obj: PriceForecast,
    exposure: ExposureBook,
    risk: RiskAppetite,
    forward_price: float,
    hedge_ratio_grid: np.ndarray | None = None,
    n_workers: int | None = None,
) -> list[dict]:
    """
    Sweep a grid of staggered hedge ratios, simulate each, and score them.

    Improvement #2: Uses ProcessPoolExecutor for parallel evaluation.
    Falls back to sequential execution if multiprocessing is unavailable
    (some cloud environments restrict fork).

    Args:
        forecast_obj: shared PriceForecast (generated ONCE, reused across all candidates)
        exposure: ExposureBook
        risk: RiskAppetite (weights + cvar_alpha + max_hedge)
        forward_price: locked-in price for hedged volumes
        hedge_ratio_grid: 1-D array of fractions to test; defaults to 11-point grid
        n_workers: number of worker processes (default None = os.cpu_count())

    Returns:
        list of dicts sorted by blended score ascending (best first)
    """
    if hedge_ratio_grid is None:
        hedge_ratio_grid = DEFAULT_HEDGE_GRID

    grid = np.asarray(hedge_ratio_grid, dtype=float)
    grid = grid[grid <= risk.max_hedge]
    if len(grid) == 0:
        raise ValueError(
            f"No hedge ratios survive max_hedge={risk.max_hedge} constraint."
        )

    # no-hedge baseline — computed once
    no_hedge_params = StrategyParams(
        strategy_type=StrategyType.STAGGERED,
        base_fraction=0.0,
    )
    no_hedge_cost = simulate_cost(
        forecast_obj, exposure, no_hedge_params, forward_price,
        cvar_alpha=risk.cvar_alpha, mode="optimized",
    )

    # Serialize to plain arrays for passing to workers
    paths_arr = np.asarray(forecast_obj.paths)
    volumes_arr = np.asarray(exposure.volumes)
    worker_args = [
        (ratio, paths_arr, volumes_arr, forward_price, risk.cvar_alpha, risk.max_hedge)
        for ratio in grid
    ]

    results = []
    parallel_used = False

    # Improvement #2: try parallel, fall back to sequential
    try:
        if n_workers is None:
            n_workers = os.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = list(executor.map(_evaluate_single_candidate, worker_args))
        parallel_used = True
        for ratio, params, cost in futures:
            score = score_policy(cost, no_hedge_cost, params, risk)
            results.append({"params": params, "cost": cost, "score": score})
    except Exception as e:
        if parallel_used:
            raise
        print(f"[scorer] Multiprocessing unavailable ({e}), falling back to sequential.")
        for ratio in grid:
            params = StrategyParams(
                strategy_type=StrategyType.STAGGERED,
                base_fraction=float(ratio),
                cap=float(risk.max_hedge),
            )
            cost = simulate_cost(
                forecast_obj, exposure, params, forward_price,
                cvar_alpha=risk.cvar_alpha, mode="optimized",
            )
            score = score_policy(cost, no_hedge_cost, params, risk)
            results.append({"params": params, "cost": cost, "score": score})

    mode_str = f"parallel (n_workers={n_workers})" if parallel_used else "sequential"
    print(f"[scorer] evaluate_candidates ran in {mode_str} mode.")

    results.sort(key=lambda r: r["score"].blended)
    return results
