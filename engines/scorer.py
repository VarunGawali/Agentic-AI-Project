"""
Scorer: evaluates and ranks hedging candidates.

BASELINE (Phase 1): sweeps a hedge-ratio grid, simulates each, scores on
  cost + CVaR (2-factor). Opportunity cost and execution risk are stubbed at 0.
UPGRADE  (Phase 3): full 4-factor normalised score + Bayesian param search.
"""

from __future__ import annotations
import numpy as np

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

    BASELINE: 2-factor (cost + λ·CVaR). Opportunity cost and execution risk
    are set to 0 until Phase 3.

    Args:
        cost: CostDistribution for the candidate policy
        no_hedge_cost: CostDistribution for the 0% hedge baseline
        params: StrategyParams of this candidate (used for execution risk proxy)
        risk: RiskAppetite with factor weights

    Returns:
        FactorScore with blended composite (lower = better)
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


def evaluate_candidates(
    forecast_obj: PriceForecast,
    exposure: ExposureBook,
    risk: RiskAppetite,
    forward_price: float,
    hedge_ratio_grid: np.ndarray | None = None,
) -> list[dict]:
    """
    Sweep a grid of staggered hedge ratios, simulate each, and score them.

    Args:
        forecast_obj: shared PriceForecast (generated ONCE, reused across all candidates)
        exposure: ExposureBook
        risk: RiskAppetite (weights + cvar_alpha + max_hedge)
        forward_price: locked-in price for hedged volumes
        hedge_ratio_grid: 1-D array of fractions to test; defaults to 11-point grid

    Returns:
        list of dicts, each containing:
            "params": StrategyParams
            "cost":   CostDistribution
            "score":  FactorScore
        sorted by blended score ascending (best first)
    """
    if hedge_ratio_grid is None:
        hedge_ratio_grid = DEFAULT_HEDGE_GRID

    # enforce max_hedge constraint
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

    results = []
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

    results.sort(key=lambda r: r["score"].blended)
    return results
