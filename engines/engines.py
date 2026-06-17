"""
Analytics engines -- the tools the agent calls. SKELETON STUBS.

Each function's signature and docstring define its contract. Phase 1 fills the
baseline logic (GBM, plain Monte Carlo, staggered, 2-factor); Phase 3 upgrades
each in place (GARCH-t, Sobol+antithetic, +trigger/+volatility, 4-factor).

A minimal baseline is implemented where it's trivial enough to make the
skeleton run end-to-end; heavier parts raise NotImplementedError with a note.
"""

from __future__ import annotations
import numpy as np

from hedging_assistant.contracts import (
    PriceHistory, PriceForecast, ExposureBook, RiskAppetite,
    StrategyType, StrategyParams, HedgingPolicy,
    CostDistribution, FactorScore,
)


# ----------------------------------------------------------------------------
# 1. FORECASTER   PriceHistory -> PriceForecast
# ----------------------------------------------------------------------------
def forecast(history: PriceHistory,
             horizon: int,
             n_paths: int = 10_000,
             seed: int | None = None) -> PriceForecast:
    """
    CONTRACT
      in : price history, horizon (periods), n_paths
      out: PriceForecast with paths shape (n_paths, horizon)

    BASELINE (Phase 1): Geometric Brownian Motion.
    UPGRADE  (Phase 3): GARCH-t (arch lib) or probabilistic ML (Darts/AutoGluon).
    """
    rng = np.random.default_rng(seed)
    log_ret = np.diff(np.log(history.prices))
    mu, sigma = log_ret.mean(), log_ret.std(ddof=1)
    s0 = history.prices[-1]
    z = rng.standard_normal((n_paths, horizon))
    increments = (mu - 0.5 * sigma**2) + sigma * z
    paths = s0 * np.exp(np.cumsum(increments, axis=1))
    return PriceForecast(paths=paths, model_name="GBM")


# ----------------------------------------------------------------------------
# 2. STRATEGY LIBRARY   StrategyParams -> per-path hedge fractions
# ----------------------------------------------------------------------------
def apply_strategy(params: StrategyParams,
                   price_path: np.ndarray,
                   history_vol: float | None = None) -> np.ndarray:
    """
    CONTRACT
      in : strategy params, ONE price path (shape (horizon,))
      out: hedge fraction chosen at each period (shape (horizon,))

    BASELINE (Phase 1): staggered only (ignores the path).
    UPGRADE  (Phase 3): trigger-based + volatility-based (path-dependent).
    """
    horizon = len(price_path)
    if params.strategy_type == StrategyType.STAGGERED:
        return np.full(horizon, params.base_fraction)

    if params.strategy_type == StrategyType.TRIGGER:
        raise NotImplementedError("Phase 3: hedge more when price crosses trigger_price")

    if params.strategy_type == StrategyType.VOLATILITY:
        raise NotImplementedError("Phase 3: hedge more when recent volatility exceeds vol_threshold")

    raise ValueError(f"unknown strategy {params.strategy_type}")


def build_policy(params: StrategyParams, forecast_obj: PriceForecast) -> HedgingPolicy:
    """Produce a representative HedgingPolicy (uses the median path for display)."""
    median_path = np.percentile(forecast_obj.paths, 50, axis=0)
    fractions = apply_strategy(params, median_path)
    desc = f"{params.base_fraction:.0%} {params.strategy_type.value}"
    return HedgingPolicy(params=params, hedge_fractions=fractions, description=desc)


# ----------------------------------------------------------------------------
# 3. MONTE CARLO EVALUATOR
#    (PriceForecast, ExposureBook, StrategyParams, forward_price) -> CostDistribution
# ----------------------------------------------------------------------------
def simulate_cost(forecast_obj: PriceForecast,
                  exposure: ExposureBook,
                  params: StrategyParams,
                  forward_price: float,
                  cvar_alpha: float = 0.95) -> CostDistribution:
    """
    CONTRACT
      in : forecast paths, exposure book, strategy params, forward price
      out: CostDistribution (costs per path + mean/p10/p50/p90/cvar)

    BASELINE (Phase 1): plain sampling over all paths.
    UPGRADE  (Phase 3): Sobol quasi-MC + antithetic variates (fewer paths).
    """
    paths = forecast_obj.paths               # (n_paths, horizon)
    volumes = exposure.volumes               # (horizon,)
    # hedge fraction per period (baseline: same across paths)
    frac = apply_strategy(params, paths[0])  # staggered ignores path content
    hedged = (frac * volumes * forward_price).sum()
    unhedged = ((1 - frac) * volumes * paths).sum(axis=1)
    total = hedged + unhedged
    return CostDistribution(costs=total, cvar_alpha=cvar_alpha)


# ----------------------------------------------------------------------------
# 4. SCORER   (CostDistribution + context) -> FactorScore
# ----------------------------------------------------------------------------
def score_policy(cost: CostDistribution,
                 no_hedge_cost: CostDistribution,
                 params: StrategyParams,
                 risk: RiskAppetite) -> FactorScore:
    """
    CONTRACT
      in : this policy's cost dist, the no-hedge cost dist (for opportunity
           cost), strategy params (for execution risk), risk appetite (weights)
      out: FactorScore (4 factors + blended composite)

    BASELINE (Phase 1): cost + lambda*cvar (other two factors = 0).
    UPGRADE  (Phase 3): full 4-factor with normalisation across factors.
    """
    cost_factor = cost.mean
    cvar_factor = cost.cvar
    # opportunity cost / execution risk: stubbed at 0 for the baseline
    opp = 0.0
    exe = 0.0
    blended = risk.w_cost * cost_factor + risk.w_cvar * cvar_factor
    return FactorScore(cost=cost_factor, cvar=cvar_factor,
                       opportunity_cost=opp, execution_risk=exe,
                       blended=blended)
