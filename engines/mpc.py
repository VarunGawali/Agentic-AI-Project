"""
Receding-horizon (Model Predictive Control) hedging — approximate dynamic programming.

The CVaR-LP produces an *open-loop* schedule: it commits f_1..f_H at t=0 using the
t=0 forecast and forward curve. MPC is the *closed-loop* version — the honest,
tractable form of dynamic programming here:

    at each period m:
        1. re-forecast volatility using prices realized so far
        2. re-anchor the forward curve at the current spot
        3. re-solve the CVaR-LP for the REMAINING horizon (m..H-1)
        4. execute (lock) only the first action f_m
        5. advance one period and repeat

This is exactly Bellman's principle solved by re-optimization instead of a stored
policy table (standard "MPC = approximate DP"). It reuses the existing CVaR-LP, so
it adds *adaptivity* without new optimization machinery.

Trade-off worth knowing (and documenting): committing the whole curve at t=0
(open-loop) locks far-month forwards at their quoted prices, capturing the full
term-structure *basis*; MPC instead hedges progressively at rolling near-forwards,
gaining *volatility adaptivity* but capturing less static basis. Which wins is an
empirical question — hence the backtest.
"""

from __future__ import annotations

import numpy as np

from hedging_assistant.contracts import ExposureBook, RiskAppetite
from hedging_assistant.engines.optimizer import optimize_cvar_lp_params


def receding_horizon_execute(
    forecast_fn,
    curve_fn,
    realized_prices: np.ndarray,
    initial_history: np.ndarray,
    exposure: ExposureBook,
    risk: RiskAppetite,
) -> dict:
    """
    Simulate closed-loop MPC execution over one realized price path.

    Parameters
    ----------
    forecast_fn : callable (hist_prices: np.ndarray, remaining: int) -> PriceForecast
        Re-forecast the remaining horizon from the price history available *now*.
    curve_fn : callable (spot: float, remaining: int) -> np.ndarray
        Build the forward curve (length `remaining`) anchored at the current spot.
    realized_prices : (H,) actual spot per procurement period (the truth path).
    initial_history : (T,) price history strictly before period 0.
    exposure, risk : standard contracts.

    Returns
    -------
    dict with executed_fractions (H,), realized_cost, and per_period detail.
    """
    realized_prices = np.asarray(realized_prices, dtype=float)
    initial_history = np.asarray(initial_history, dtype=float)
    volumes = np.asarray(exposure.volumes, dtype=float)
    H = len(realized_prices)

    if len(volumes) != H:
        raise ValueError("exposure horizon must match realized_prices length")

    executed = np.zeros(H, dtype=float)
    per_period = []
    total_cost = 0.0

    for m in range(H):
        hist = np.concatenate([initial_history, realized_prices[:m]])
        spot = float(hist[-1])
        remaining = H - m

        fc = forecast_fn(hist, remaining)
        curve = np.asarray(curve_fn(spot, remaining), dtype=float)
        sub_exposure = ExposureBook(volumes=volumes[m:])

        try:
            params = optimize_cvar_lp_params(
                forecast_obj=fc,
                exposure=sub_exposure,
                forward_price=curve,
                cvar_alpha=risk.cvar_alpha,
                cost_weight=risk.w_cost,
                cvar_weight=risk.w_cvar,
                opportunity_weight=risk.w_opportunity,
                execution_weight=risk.w_execution,
                max_hedge=risk.max_hedge,
            )
            f0 = float(np.asarray(params.fixed_fractions, dtype=float)[0])
        except Exception:
            f0 = 0.0

        f0 = float(np.clip(f0, 0.0, risk.max_hedge))
        F0 = float(curve[0])
        cost_m = f0 * volumes[m] * F0 + (1.0 - f0) * volumes[m] * float(realized_prices[m])

        executed[m] = f0
        total_cost += cost_m
        per_period.append(
            {"period": m, "hedge": f0, "forward": F0, "spot": float(realized_prices[m])}
        )

    return {
        "executed_fractions": executed,
        "realized_cost": float(total_cost),
        "per_period": per_period,
    }
