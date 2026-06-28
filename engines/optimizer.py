"""
Phase 3 optimizer: CVaR-LP + Optuna hyperparameter search.

Features:
    1. optimize_cvar_lp()
        Rockafellar-Uryasev CVaR LP using cvxpy.
        Produces optimal per-period hedge fractions.

    2. cvar_lp_params_from_fractions()
        Converts LP hedge schedule into StrategyParams(strategy_type=CVAR_LP).

    3. tune_strategy_params()
        Optuna TPE search over non-convex strategy parameters:
            - base_fraction
            - cap
            - trigger_threshold
            - trigger_fraction
            - vol_scale_k

    4. strategy_params_from_optuna_result()
        Converts Optuna result dict into StrategyParams.

Notes:
    - CVaR-LP requires cvxpy.
    - Optuna tuning requires optuna.
    - CVAR_LP strategy requires:
        StrategyType.CVAR_LP
        StrategyParams.fixed_fractions
      to be added in contracts.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hedging_assistant.contracts import (
    PriceForecast,
    ExposureBook,
    RiskAppetite,
    StrategyParams,
    StrategyType,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _resolve_forward_curve(
    forward_price,
    horizon: int,
) -> np.ndarray:
    """
    Convert scalar/list/ForwardCurve-like object into a forward curve array.

    Supported:
        - scalar float
        - list / np.ndarray of shape (horizon,)
        - object with .prices
    """

    if np.isscalar(forward_price):
        if float(forward_price) <= 0:
            raise ValueError(f"forward_price must be positive; got {forward_price}")

        return np.full(horizon, float(forward_price), dtype=float)

    if hasattr(forward_price, "prices"):
        fwd = np.asarray(forward_price.prices, dtype=float)
    else:
        fwd = np.asarray(forward_price, dtype=float)

    if fwd.ndim != 1:
        raise ValueError("forward curve must be a 1D array")

    if len(fwd) != horizon:
        raise ValueError(
            f"forward curve length {len(fwd)} does not match horizon {horizon}"
        )

    if np.any(fwd <= 0):
        raise ValueError("forward curve prices must all be positive")

    return fwd


# ---------------------------------------------------------------------------
# 1. Rockafellar-Uryasev CVaR-LP
# ---------------------------------------------------------------------------

def optimize_cvar_lp(
    forecast_obj: PriceForecast,
    exposure: ExposureBook,
    forward_price,
    cvar_alpha: float = 0.95,
    cost_weight: float = 1.0,
    cvar_weight: float = 1.0,
    opportunity_weight: float = 0.0,
    execution_weight: float = 0.0,
    execution_cost_per_barrel: float = 0.05,
    max_hedge: float = 1.0,
    solver: str | None = None,
) -> np.ndarray:
    """
    Solve for optimal per-period hedge fractions using a 4-factor LP.

    Decision variables:
        f : hedge fraction per period, shape (H,)
        z : VaR threshold (CVaR linearization)
        u : tail excess per scenario, shape (N,)   [CVaR slack]
        r : regret per scenario, shape (N,)        [opportunity cost slack]

    Objective:
        minimize  cost_weight      * mean_cost
                + cvar_weight      * CVaR_alpha
                + opportunity_weight * mean_regret
                + execution_weight * execution_risk

    CVaR linearization (Rockafellar-Uryasev):
        CVaR_alpha = z + 1 / ((1 - alpha) * N) * sum(u)
        u_i >= total_cost_i - z
        u_i >= 0

    Opportunity cost linearization:
        regret_i  = max(total_cost_i - no_hedge_cost_i, 0)
        r_i >= total_cost_i - no_hedge_cost_i
        r_i >= 0
        mean_regret = (1/N) * sum(r)

    Execution risk (linear in f):
        execution_risk = sum_t(f_t * volume_t) * execution_cost_per_barrel

    total_cost_i(f):
        sum_t volume_t * [f_t * forward_t + (1 - f_t) * spot_i_t]

    Returns:
        np.ndarray of shape (H,) with hedge fractions in [0, max_hedge].
    """

    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "cvxpy is required for CVaR-LP. Install with: uv pip install cvxpy"
        ) from exc

    paths = np.asarray(forecast_obj.paths, dtype=float)
    volumes = np.asarray(exposure.volumes, dtype=float)

    if paths.ndim != 2:
        raise ValueError("forecast_obj.paths must be 2D: (n_paths, horizon)")

    if volumes.ndim != 1:
        raise ValueError("exposure.volumes must be 1D")

    n_paths, horizon = paths.shape

    if n_paths == 0:
        raise ValueError("forecast_obj.paths cannot have zero paths")

    if horizon == 0:
        raise ValueError("forecast horizon cannot be zero")

    if len(volumes) != horizon:
        raise ValueError(
            f"Exposure length {len(volumes)} does not match forecast horizon {horizon}"
        )

    if np.any(paths <= 0):
        raise ValueError("forecast paths must contain positive prices")

    if np.any(volumes < 0):
        raise ValueError("exposure volumes cannot be negative")

    if not 0 < cvar_alpha < 1:
        raise ValueError("cvar_alpha must be in (0, 1)")

    if cost_weight < 0:
        raise ValueError("cost_weight cannot be negative")

    if cvar_weight < 0:
        raise ValueError("cvar_weight cannot be negative")

    if opportunity_weight < 0:
        raise ValueError("opportunity_weight cannot be negative")

    if execution_weight < 0:
        raise ValueError("execution_weight cannot be negative")

    if execution_cost_per_barrel < 0:
        raise ValueError("execution_cost_per_barrel cannot be negative")

    if not 0 <= max_hedge <= 1:
        raise ValueError("max_hedge must be between 0 and 1")

    fwd = _resolve_forward_curve(
        forward_price=forward_price,
        horizon=horizon,
    )

    # Decision variables
    f = cp.Variable(horizon, name="hedge_fractions")
    z = cp.Variable(name="var_threshold")
    u = cp.Variable(n_paths, name="tail_excess")   # CVaR slack
    r = cp.Variable(n_paths, name="regret")         # opportunity cost slack

    # total_cost_i(f)
    # = sum_t volume_t * [f_t * fwd_t + (1 - f_t) * spot_i_t]
    #
    # = sum_t volume_t * spot_i_t
    #   + sum_t f_t * volume_t * (fwd_t - spot_i_t)
    #
    # = b_i + A_i @ f

    vol_fwd = volumes * fwd                  # shape (H,)
    vol_path = volumes[None, :] * paths      # shape (N, H)

    A = vol_fwd[None, :] - vol_path          # shape (N, H)
    b = vol_path.sum(axis=1)                 # shape (N,)

    total_costs = A @ f + b                  # affine expression, shape (N,)

    # No-hedge cost per scenario (constant — no decision variable involved)
    no_hedge_costs = vol_path.sum(axis=1)    # shape (N,) — pure spot exposure

    mean_cost = cp.sum(total_costs) / n_paths

    cvar_cost = z + (
        1.0 / ((1.0 - cvar_alpha) * n_paths)
    ) * cp.sum(u)

    # Opportunity cost: mean regret across scenarios where hedging
    # turned out more expensive than doing nothing.
    # r_i >= total_cost_i - no_hedge_cost_i, r_i >= 0
    mean_regret = cp.sum(r) / n_paths

    # Execution risk: total hedged volume × cost per barrel (linear in f)
    execution_risk = float(execution_cost_per_barrel) * (volumes @ f)

    objective = cp.Minimize(
        cost_weight      * mean_cost
        + cvar_weight      * cvar_cost
        + opportunity_weight * mean_regret
        + execution_weight   * execution_risk
    )

    constraints = [
        f >= 0,
        f <= max_hedge,
        u >= 0,
        u >= total_costs - z,
        r >= 0,
        r >= total_costs - no_hedge_costs,
    ]

    problem = cp.Problem(objective, constraints)

    candidate_solvers = []

    if solver:
        candidate_solvers.append(solver)
    else:
        # CLARABEL is usually available with modern cvxpy.
        # ECOS/SCS are fallback options if installed.
        candidate_solvers = ["CLARABEL", "ECOS", "SCS"]

    last_error = None

    for candidate_solver in candidate_solvers:
        try:
            problem.solve(
                solver=candidate_solver,
                verbose=False,
            )

            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                break

        except Exception as exc:
            last_error = exc

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(
            f"CVaR-LP solver failed. status={problem.status}, error={last_error}"
        )

    if f.value is None:
        raise RuntimeError("CVaR-LP returned no hedge fraction solution")

    hedge_fractions = np.asarray(f.value, dtype=float)
    hedge_fractions = np.clip(hedge_fractions, 0.0, max_hedge)

    return hedge_fractions


def cvar_lp_params_from_fractions(
    hedge_fractions: np.ndarray,
    max_hedge: float = 1.0,
) -> StrategyParams:
    """
    Convert CVaR-LP hedge fractions into StrategyParams.

    This allows the LP output to flow through:
        - apply_strategy()
        - simulate_cost()
        - build_policy()
        - CandidateRecord
        - dashboard hedge schedule
    """

    hedge_fractions = np.asarray(hedge_fractions, dtype=float)

    if hedge_fractions.ndim != 1:
        raise ValueError("hedge_fractions must be a 1D array")

    if len(hedge_fractions) == 0:
        raise ValueError("hedge_fractions cannot be empty")

    if not 0 <= max_hedge <= 1:
        raise ValueError("max_hedge must be between 0 and 1")

    if np.any((hedge_fractions < 0) | (hedge_fractions > 1)):
        raise ValueError("hedge_fractions must be between 0 and 1")

    hedge_fractions = np.clip(hedge_fractions, 0.0, max_hedge)

    return StrategyParams(
        strategy_type=StrategyType.CVAR_LP,
        base_fraction=float(np.mean(hedge_fractions)),
        fixed_fractions=hedge_fractions,
        cap=max_hedge,
    )


def optimize_cvar_lp_params(
    forecast_obj: PriceForecast,
    exposure: ExposureBook,
    forward_price,
    cvar_alpha: float = 0.95,
    cost_weight: float = 1.0,
    cvar_weight: float = 1.0,
    opportunity_weight: float = 0.0,
    execution_weight: float = 0.0,
    execution_cost_per_barrel: float = 0.05,
    max_hedge: float = 1.0,
    solver: str | None = None,
) -> StrategyParams:
    """
    Convenience wrapper:
        optimize 4-factor LP fractions
        convert them directly into StrategyParams(strategy_type=CVAR_LP)
    """

    hedge_fractions = optimize_cvar_lp(
        forecast_obj=forecast_obj,
        exposure=exposure,
        forward_price=forward_price,
        cvar_alpha=cvar_alpha,
        cost_weight=cost_weight,
        cvar_weight=cvar_weight,
        opportunity_weight=opportunity_weight,
        execution_weight=execution_weight,
        execution_cost_per_barrel=execution_cost_per_barrel,
        max_hedge=max_hedge,
        solver=solver,
    )

    return cvar_lp_params_from_fractions(
        hedge_fractions=hedge_fractions,
        max_hedge=max_hedge,
    )


# ---------------------------------------------------------------------------
# 2. Optuna hyperparameter search
# ---------------------------------------------------------------------------

def tune_strategy_params(
    forecast_obj: PriceForecast,
    exposure: ExposureBook,
    risk: RiskAppetite,
    forward_price,
    n_trials: int = 40,
    seed: int = 42,
    strategy_type: StrategyType = StrategyType.HYBRID,
) -> dict[str, Any]:
    """
    Tune non-convex strategy parameters using Optuna.

    Search space:
        base_fraction
        cap
        trigger_threshold
        trigger_fraction
        vol_scale_k

    Intended for:
        TRIGGER
        VOLATILITY
        HYBRID

    Returns:
        Dictionary containing the best parameters and best score.
    """

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError(
            "optuna is required for strategy tuning. Install with: uv pip install optuna"
        ) from exc

    from engines.cost_simulator import simulate_cost
    from engines.scorer import score_policy

    if not isinstance(strategy_type, StrategyType):
        strategy_type = StrategyType(strategy_type)

    if strategy_type not in {
        StrategyType.TRIGGER,
        StrategyType.VOLATILITY,
        StrategyType.HYBRID,
        StrategyType.STAGGERED,
    }:
        raise ValueError(
            "Optuna tuning supports STAGGERED, TRIGGER, VOLATILITY, and HYBRID."
        )

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    if not 0 <= risk.max_hedge <= 1:
        raise ValueError("risk.max_hedge must be between 0 and 1")

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

    def objective(trial: "optuna.Trial") -> float:
        base_fraction = trial.suggest_float(
            "base_fraction",
            0.0,
            risk.max_hedge,
        )

        cap = trial.suggest_float(
            "cap",
            max(base_fraction, 0.01),
            risk.max_hedge,
        )

        trigger_threshold = trial.suggest_float(
            "trigger_threshold",
            0.95,
            1.15,
        )

        trigger_fraction = trial.suggest_float(
            "trigger_fraction",
            base_fraction,
            risk.max_hedge,
        )

        vol_scale_k = trial.suggest_float(
            "vol_scale_k",
            0.0,
            3.0,
        )

        params = StrategyParams(
            strategy_type=strategy_type,
            base_fraction=base_fraction,
            cap=cap,
            trigger_threshold=trigger_threshold,
            trigger_fraction=trigger_fraction,
            vol_scale_k=vol_scale_k,
            ma_window=5,
        )

        cost = simulate_cost(
            forecast_obj=forecast_obj,
            exposure=exposure,
            params=params,
            forward_price=forward_price,
            cvar_alpha=risk.cvar_alpha,
            mode="accurate",
            compute_ci=False,
        )

        score = score_policy(
            cost=cost,
            no_hedge_cost=no_hedge_cost,
            params=params,
            risk=risk,
        )

        return float(score.blended)

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = dict(study.best_params)
    best["best_score"] = float(study.best_value)
    best["strategy_type"] = strategy_type.value
    best["n_trials"] = int(n_trials)

    print(
        f"[optimizer] Optuna {strategy_type.value}: "
        f"{n_trials} trials -> "
        f"base_fraction={best['base_fraction']:.2%}, "
        f"cap={best['cap']:.2%}, "
        f"score={best['best_score']:,.2f}"
    )

    return best


def strategy_params_from_optuna_result(
    result: dict[str, Any],
) -> StrategyParams:
    """
    Convert Optuna result dictionary into StrategyParams.
    """

    strategy_type = StrategyType(result.get("strategy_type", "hybrid"))

    return StrategyParams(
        strategy_type=strategy_type,
        base_fraction=float(result["base_fraction"]),
        cap=float(result["cap"]),
        trigger_threshold=float(result.get("trigger_threshold", 1.05)),
        trigger_fraction=float(
            result.get("trigger_fraction", result["base_fraction"])
        ),
        vol_scale_k=float(result.get("vol_scale_k", 1.0)),
        ma_window=int(result.get("ma_window", 5)),
    )