"""
Scorer: evaluates and ranks hedging candidates.

Phase 1:
    - Sweep staggered hedge ratios
    - Simulate each policy
    - Score using expected cost + CVaR

Phase 3:
    - Four-factor scoring:
        Score =
            w_cost * Expected Cost
          + w_cvar * CVaR
          + w_opportunity * Opportunity Cost
          + w_execution * Execution Risk

    - Opportunity cost:
        Mean regret vs no-hedge when strategy is more expensive than no hedge.

    - Execution risk:
        Simple transaction/operational-cost proxy based on hedged volume.

    - Supports rule-based candidates plus optimizer-generated strategies
      such as CVAR_LP.
"""

from __future__ import annotations

import concurrent.futures
import numpy as np

from hedging_assistant.contracts import (
    PriceForecast,
    ExposureBook,
    RiskAppetite,
    StrategyType,
    StrategyParams,
    CostDistribution,
    FactorScore,
)

from hedging_assistant.engines.cost_simulator import simulate_cost
from hedging_assistant.engines.strategy_library import (
    apply_strategy,
    generate_staggered_candidates,
)


DEFAULT_HEDGE_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Factor helpers
# ---------------------------------------------------------------------------

def compute_opportunity_cost(
    cost: CostDistribution,
    no_hedge_cost: CostDistribution,
) -> float:
    """
    Compute opportunity cost / regret versus no-hedge.

    Interpretation:
        If the hedge strategy costs more than no hedge in a scenario,
        that excess is treated as missed upside.

    Formula:
        opportunity_cost =
            mean(max(strategy_cost_i - no_hedge_cost_i, 0))
    """

    strategy_costs = np.asarray(cost.costs, dtype=float)
    baseline_costs = np.asarray(no_hedge_cost.costs, dtype=float)

    if strategy_costs.shape != baseline_costs.shape:
        raise ValueError(
            "cost.costs and no_hedge_cost.costs must have the same shape "
            "to compute opportunity cost."
        )

    regret = np.maximum(strategy_costs - baseline_costs, 0.0)

    return float(regret.mean())


def estimate_hedge_fractions_for_execution(
    params: StrategyParams,
    horizon: int,
    forecast_obj: PriceForecast | None = None,
) -> np.ndarray:
    """
    Estimate representative hedge schedule for execution-risk calculation.

    For path-independent strategies:
        Uses fixed/base schedule directly.

    For path-dependent strategies:
        Uses median forecast path as representative schedule.

    This avoids running execution-risk calculation path-by-path, while still
    giving a reasonable operational burden proxy.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    # CVAR_LP / fixed schedule
    if params.strategy_type == StrategyType.CVAR_LP:
        if params.fixed_fractions is None:
            raise ValueError("CVAR_LP requires fixed_fractions.")

        fractions = np.asarray(params.fixed_fractions, dtype=float)

        if len(fractions) != horizon:
            raise ValueError(
                f"CVAR_LP fixed_fractions length {len(fractions)} does not "
                f"match horizon {horizon}"
            )

        return np.clip(fractions, 0.0, params.cap)

    # STAGGERED schedule
    if params.strategy_type == StrategyType.STAGGERED:
        return np.full(
            horizon,
            min(float(params.base_fraction), float(params.cap)),
            dtype=float,
        )

    # Path-dependent strategies use median forecast path.
    if forecast_obj is not None:
        paths = np.asarray(forecast_obj.paths, dtype=float)

        if paths.ndim != 2:
            raise ValueError("forecast_obj.paths must be 2D")

        if paths.shape[1] != horizon:
            raise ValueError(
                f"forecast horizon={paths.shape[1]} does not match expected "
                f"horizon={horizon}"
            )

        median_path = np.percentile(paths, 50, axis=0)

        return apply_strategy(
            params=params,
            price_path=median_path,
        )

    # Fallback if no forecast object is available.
    return np.full(
        horizon,
        min(float(params.base_fraction), float(params.cap)),
        dtype=float,
    )


def compute_execution_risk(
    params: StrategyParams,
    exposure: ExposureBook | None = None,
    forecast_obj: PriceForecast | None = None,
    execution_cost_per_barrel: float = 0.05,
) -> float:
    """
    Compute execution risk as a simple transaction/operational burden proxy.

    Formula:
        execution_risk =
            sum_t(hedge_fraction_t * volume_t) * execution_cost_per_barrel

    Units:
        dollars, assuming execution_cost_per_barrel is USD/bbl.

    Why this simple proxy:
        - Larger hedge volumes are operationally harder.
        - Larger hedge volumes have more brokerage/spread/slippage exposure.
        - It gives the scorer a way to penalize over-hedging.

    Later versions can add:
        - turnover penalty
        - liquidity penalty
        - concentration penalty
        - complexity penalty by strategy type
    """

    if execution_cost_per_barrel < 0:
        raise ValueError("execution_cost_per_barrel cannot be negative")

    if exposure is None:
        # Backward-compatible fallback.
        # No exposure means we cannot compute dollar execution burden.
        return 0.0

    volumes = np.asarray(exposure.volumes, dtype=float)

    if volumes.ndim != 1:
        raise ValueError("exposure.volumes must be 1D")

    if np.any(volumes < 0):
        raise ValueError("exposure volumes cannot be negative")

    horizon = len(volumes)

    fractions = estimate_hedge_fractions_for_execution(
        params=params,
        horizon=horizon,
        forecast_obj=forecast_obj,
    )

    if len(fractions) != horizon:
        raise ValueError(
            f"execution hedge schedule length {len(fractions)} does not match "
            f"exposure horizon {horizon}"
        )

    if np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("execution hedge fractions must be between 0 and 1")

    hedged_volume = float(np.sum(fractions * volumes))

    return hedged_volume * float(execution_cost_per_barrel)


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def score_policy(
    cost: CostDistribution,
    no_hedge_cost: CostDistribution,
    params: StrategyParams,
    risk: RiskAppetite,
    exposure: ExposureBook | None = None,
    forecast_obj: PriceForecast | None = None,
    execution_cost_per_barrel: float = 0.05,
) -> FactorScore:
    """
    Score one hedge policy using the four-factor objective.

    Formula:
        Score =
            w_cost * Expected Cost
          + w_cvar * CVaR
          + w_opportunity * Opportunity Cost
          + w_execution * Execution Risk

    Lower blended score is better.

    NOTE: the `blended` set here is a provisional raw-dollar weighted sum used
    only as a fallback ordering for callers that do not normalize. The
    authoritative, magnitude-invariant ranking is produced by `blend_scores()`,
    which min-max normalizes each factor across the candidate pool. Whenever a
    full pool is available (the agent workflow, the API) call `blend_scores()`
    so no single factor dominates purely because of its dollar scale.
    """

    cost_factor = float(cost.mean)
    cvar_factor = float(cost.cvar)

    opportunity_cost = compute_opportunity_cost(
        cost=cost,
        no_hedge_cost=no_hedge_cost,
    )

    execution_risk = compute_execution_risk(
        params=params,
        exposure=exposure,
        forecast_obj=forecast_obj,
        execution_cost_per_barrel=execution_cost_per_barrel,
    )

    blended = (
        risk.w_cost * cost_factor
        + risk.w_cvar * cvar_factor
        + risk.w_opportunity * opportunity_cost
        + risk.w_execution * execution_risk
    )

    return FactorScore(
        cost=cost_factor,
        cvar=cvar_factor,
        opportunity_cost=opportunity_cost,
        execution_risk=execution_risk,
        blended=float(blended),
    )


# ---------------------------------------------------------------------------
# Pool-level normalization + blend
# ---------------------------------------------------------------------------

def blend_scores(
    results: list[dict],
    risk: RiskAppetite,
    eps: float = 1e-9,
) -> list[dict]:
    """
    Normalize the four factors across the candidate pool, then blend.

    Each raw factor (cost, cvar, opportunity_cost, execution_risk) is min-max
    scaled to [0, 1] over the whole candidate set, so no factor dominates the
    blended score purely because of dollar magnitude. The blended score becomes
    a unitless multi-criteria index in [0, sum(weights)] — lower is better:

        blended = w_cost*cost_n + w_cvar*cvar_n
                + w_opportunity*opp_n + w_execution*exec_n

    A factor that is constant across all candidates (zero spread) contributes 0,
    so it cannot tip the decision either way.

    Mutates each item["score"] in place (fills the *_norm fields and `blended`)
    and returns the pool sorted ascending by blended.
    """
    if not results:
        return results

    factor_attrs = ("cost", "cvar", "opportunity_cost", "execution_risk")
    norm_attrs = {
        "cost": "cost_norm",
        "cvar": "cvar_norm",
        "opportunity_cost": "opportunity_norm",
        "execution_risk": "execution_norm",
    }
    weight_for = {
        "cost": risk.w_cost,
        "cvar": risk.w_cvar,
        "opportunity_cost": risk.w_opportunity,
        "execution_risk": risk.w_execution,
    }

    raw = {
        attr: np.array(
            [float(getattr(item["score"], attr)) for item in results],
            dtype=float,
        )
        for attr in factor_attrs
    }

    norm = {}
    for attr in factor_attrs:
        vals = raw[attr]
        lo = float(vals.min())
        hi = float(vals.max())
        span = hi - lo
        norm[attr] = (vals - lo) / span if span > eps else np.zeros_like(vals)

    for i, item in enumerate(results):
        score = item["score"]
        for attr in factor_attrs:
            setattr(score, norm_attrs[attr], float(norm[attr][i]))
        score.blended = float(
            sum(weight_for[attr] * norm[attr][i] for attr in factor_attrs)
        )

    return sorted(results, key=lambda item: item["score"].blended)


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_candidates(
    forecast_obj: PriceForecast,
    exposure: ExposureBook,
    risk: RiskAppetite,
    forward_price,
    hedge_ratio_grid: list[float] | np.ndarray | None = None,
    mode: str = "optimized",
    compute_ci: bool = False,
    candidates: list[StrategyParams] | None = None,
    execution_cost_per_barrel: float = 0.05,
    no_hedge_cost: "CostDistribution | None" = None,
    n_jobs: int = 4,
    price_history: "np.ndarray | None" = None,
    long_run_vol: "float | None" = None,
    normalize: bool = False,
) -> list[dict]:
    """
    Evaluate hedge candidates and return ranked results.

    If candidates is provided:
        Evaluates those StrategyParams directly.

    If candidates is None:
        Generates staggered candidates from hedge_ratio_grid.

    Returns:
        list of dictionaries:
            {
                "hedge_ratio": float,
                "strategy_type": str,
                "params": StrategyParams,
                "cost": CostDistribution,
                "score": FactorScore,
            }
    """

    if not 0 <= risk.max_hedge <= 1:
        raise ValueError("risk.max_hedge must be between 0 and 1")

    # ---------------------------------------------------------
    # 1. No-hedge baseline (skip if caller already computed it)
    # ---------------------------------------------------------

    if no_hedge_cost is None:
        no_hedge_params = StrategyParams(
            strategy_type=StrategyType.STAGGERED,
            base_fraction=0.0,
            cap=risk.max_hedge,
        )

        no_hedge_cost = simulate_cost(
            forecast_obj=forecast_obj,
            exposure=exposure,
            params=no_hedge_params,
            forward_price=forward_price,
            cvar_alpha=risk.cvar_alpha,
            mode="optimized",
            compute_ci=False,
        )

    # ---------------------------------------------------------
    # 2. Generate candidates if not supplied
    # ---------------------------------------------------------

    if candidates is None:
        if hedge_ratio_grid is None:
            hedge_ratio_grid = DEFAULT_HEDGE_GRID

        hedge_ratio_grid = np.asarray(hedge_ratio_grid, dtype=float)

        if hedge_ratio_grid.ndim != 1:
            raise ValueError("hedge_ratio_grid must be a 1D list or array")

        if np.any((hedge_ratio_grid < 0) | (hedge_ratio_grid > 1)):
            raise ValueError("hedge ratios must be between 0 and 1")

        valid_ratios = [
            float(ratio)
            for ratio in hedge_ratio_grid
            if ratio <= risk.max_hedge
        ]

        if not valid_ratios:
            raise ValueError(
                f"No hedge ratios remain after applying max_hedge={risk.max_hedge}"
            )

        candidates = generate_staggered_candidates(
            fractions=valid_ratios,
            cap=risk.max_hedge,
        )

    # ---------------------------------------------------------
    # 3. Simulate + score candidates
    # ---------------------------------------------------------

    def _eval_one(params: StrategyParams) -> dict:
        cost_dist = simulate_cost(
            forecast_obj=forecast_obj,
            exposure=exposure,
            params=params,
            forward_price=forward_price,
            cvar_alpha=risk.cvar_alpha,
            mode=mode,
            compute_ci=compute_ci,
            price_history=price_history,
            long_run_vol=long_run_vol,
        )

        score = score_policy(
            cost=cost_dist,
            no_hedge_cost=no_hedge_cost,
            params=params,
            risk=risk,
            exposure=exposure,
            forecast_obj=forecast_obj,
            execution_cost_per_barrel=execution_cost_per_barrel,
        )

        return {
            "hedge_ratio": float(params.base_fraction),
            "strategy_type": params.strategy_type.value,
            "params": params,
            "cost": cost_dist,
            "score": score,
        }

    workers = max(1, min(n_jobs, len(candidates)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_eval_one, candidates))

    if normalize:
        # Single-call pool == full candidate set, so normalize here.
        # In the fan-out agent workflow this is left False and blend_scores()
        # is applied once after the per-family results are merged.
        return blend_scores(results, risk)

    results.sort(key=lambda item: item["score"].blended)

    return results


# ---------------------------------------------------------------------------
# Backward-compatible helper
# ---------------------------------------------------------------------------

def find_best_staggered_hedge(
    forecast_obj: PriceForecast,
    exposure: ExposureBook,
    forward_price,
    risk: RiskAppetite,
    hedge_ratios: list[float] | None = None,
):
    """
    Backward-compatible helper.

    Returns:
        best, results

    best:
        dictionary for lowest-score policy

    results:
        all candidates sorted by score ascending
    """

    results = evaluate_candidates(
        forecast_obj=forecast_obj,
        exposure=exposure,
        risk=risk,
        forward_price=forward_price,
        hedge_ratio_grid=hedge_ratios,
        mode="optimized",
        compute_ci=False,
    )

    best = results[0]

    return best, results