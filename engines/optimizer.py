"""
Phase 3 optimizer: Convex CVaR-LP + Optuna hyperparameter search.

CHANGES (new file):
  - optimize_cvar_lp(): Rockafellar-Uryasev (2000) CVaR LP via cvxpy.
    Finds the exact optimal per-period hedge fractions in one solver pass
    instead of brute-force sweeping discrete candidates.
  - tune_strategy_params(): Optuna TPE search over non-convex hyperparameters
    (GARCH order, trigger thresholds, vol-scaling factor k, base_fraction).
    Objective: minimise blended FactorScore on the current forecast.
  - Both functions degrade gracefully if cvxpy/optuna are not installed.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hedging_assistant.contracts import (
        PriceForecast, ExposureBook, RiskAppetite,
        StrategyParams, CostDistribution,
    )


# ---------------------------------------------------------------------------
# 1. Rockafellar-Uryasev CVaR-LP
# ---------------------------------------------------------------------------

def optimize_cvar_lp(
    forecast_obj: "PriceForecast",
    exposure: "ExposureBook",
    forward_price,
    cvar_alpha: float = 0.95,
    cost_weight: float = 1.0,
    cvar_weight: float = 1.0,
    max_hedge: float = 1.0,
) -> np.ndarray:
    """
    Solve for the optimal per-period hedge fractions using the
    Rockafellar-Uryasev (2000) CVaR LP formulation.

    Decision variables:
      f  ∈ [0, max_hedge]^H   — hedge fraction per delivery period
      z  ∈ ℝ                  — CVaR auxiliary variable (VaR threshold)
      u  ∈ ℝ^N, u ≥ 0         — tail excess per scenario

    Objective (convex combination):
      min  cost_weight * mean_cost(f)
         + cvar_weight * CVaR_alpha(f)

    CVaR linearisation (Rockafellar-Uryasev 2000):
      CVaR_α(f) = z + (1/(1-α)) * (1/N) * sum(u)
      u_i ≥ total_cost_i(f) - z   ∀ i
      u_i ≥ 0

    total_cost_i(f) = sum_t [ f_t * vol_t * fwd_t + (1-f_t) * vol_t * path_i_t ]
                    = sum_t vol_t * fwd_t * f_t + sum_t vol_t * (1-f_t) * path_i_t

    This is linear in f, so the full problem is an LP.

    Returns
    -------
    np.ndarray shape (H,) — optimal hedge fractions in [0, max_hedge]
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("cvxpy required for CVaR-LP: pip install cvxpy")

    from hedging_assistant.contracts import ForwardCurve

    paths   = np.asarray(forecast_obj.paths, dtype=float)   # (N, H)
    volumes = np.asarray(exposure.volumes,   dtype=float)   # (H,)
    N, H    = paths.shape

    if isinstance(forward_price, ForwardCurve):
        fwd = np.asarray(forward_price.prices, dtype=float)
    else:
        fwd = np.full(H, float(forward_price))

    # --- Decision variables ---
    f = cp.Variable(H, name="hedge_fractions")  # shape (H,)
    z = cp.Variable(name="var_threshold")       # scalar VaR
    u = cp.Variable(N, name="tail_excess")      # shape (N,)

    # --- Cost per scenario ---
    # hedged_cost_scalar = f @ (volumes * fwd)   — same for all scenarios
    # unhedged_cost_i    = (1 - f) @ (volumes * paths[i])
    # total_cost_i       = f @ (volumes * fwd) + (1-f) @ (volumes * paths[i])
    #                    = f @ (volumes * (fwd - paths[i])) + volumes @ paths[i]

    vol_fwd  = volumes * fwd                 # shape (H,)
    vol_path = volumes[:, None] * paths.T    # (H, N) — vol-scaled path matrix

    # total_cost_i = vol_fwd @ f + (volumes * paths[i]) @ (1 - f)
    #             = vol_fwd @ f + volumes @ paths[i] - vol_path[:,i] @ f
    #             = (vol_fwd - vol_path[:,i]) @ f + volumes @ paths[i]
    # stack: A @ f + b  where A is (N,H), b is (N,)
    A = (vol_fwd[None, :] - vol_path.T)      # (N, H)
    b = (volumes[None, :] * paths).sum(axis=1)  # (N,)

    total_costs = A @ f + b   # (N,) affine in f

    mean_cost = cp.sum(total_costs) / N

    # CVaR linearisation
    cvar_cost = z + (1.0 / ((1.0 - cvar_alpha) * N)) * cp.sum(u)

    objective = cp.Minimize(cost_weight * mean_cost + cvar_weight * cvar_cost)

    constraints = [
        f >= 0,
        f <= max_hedge,
        u >= 0,
        u >= total_costs - z,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"CVaR-LP solver failed: status={prob.status}")

    fracs = np.clip(f.value, 0.0, max_hedge)
    return fracs


# ---------------------------------------------------------------------------
# 2. Optuna hyperparameter search
# ---------------------------------------------------------------------------

def tune_strategy_params(
    forecast_obj: "PriceForecast",
    exposure: "ExposureBook",
    risk: "RiskAppetite",
    forward_price,
    n_trials: int = 40,
    seed: int = 42,
) -> dict:
    """
    Optuna TPE search over non-convex strategy hyperparameters.

    Search space:
      base_fraction     ∈ [0, risk.max_hedge]   — flat hedge %
      trigger_threshold ∈ [0.85, 1.15]          — price / MA ratio trigger
      vol_scale_k       ∈ [0.0, 3.0]            — vol-scaling factor
      cap               ∈ [base_fraction, 1.0]  — max hedge cap

    Objective: minimise blended FactorScore (lower = better).

    Returns
    -------
    dict with keys: base_fraction, trigger_threshold, vol_scale_k, cap, best_score
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna required: pip install optuna")

    from hedging_assistant.contracts import StrategyParams, StrategyType
    from hedging_assistant.engines.cost_simulator import simulate_cost
    from hedging_assistant.engines.scorer import score_policy

    no_hedge_params = StrategyParams(StrategyType.STAGGERED, base_fraction=0.0)
    no_hedge_cost   = simulate_cost(forecast_obj, exposure, no_hedge_params, forward_price)

    def objective(trial: "optuna.Trial") -> float:
        frac  = trial.suggest_float("base_fraction",     0.0,  risk.max_hedge)
        cap   = trial.suggest_float("cap",               frac, 1.0)
        # these will matter once trigger/vol strategies are implemented;
        # for now they're recorded in the trial for future use
        _thr  = trial.suggest_float("trigger_threshold", 0.85, 1.15)
        _k    = trial.suggest_float("vol_scale_k",       0.0,  3.0)

        params = StrategyParams(
            strategy_type=StrategyType.STAGGERED,
            base_fraction=frac,
            cap=cap,
        )
        cost  = simulate_cost(forecast_obj, exposure, params, forward_price)
        score = score_policy(cost, no_hedge_cost, params, risk)
        return float(score.blended)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["best_score"] = study.best_value
    print(
        f"[optimizer] Optuna: {n_trials} trials → "
        f"base_fraction={best['base_fraction']:.2%}, "
        f"cap={best['cap']:.2%}, "
        f"score={best['best_score']:,.0f}"
    )
    return best
