"""
Walk-forward backtester -- VALIDATION HARNESS (offline).

Per the latest scoping decision, the backtester is NOT part of the live
decision flow. It is an offline tool used to VALIDATE AND QUANTIFY UPGRADES:
run it before and after a change (e.g. GBM -> GARCH) to measure whether the
change actually improved out-of-sample performance.

It wraps the whole agent+engine pipeline and replays history with no
look-ahead. SKELETON: signature + contract only; logic filled in Phase 1.
"""

from __future__ import annotations
import numpy as np

from hedging_assistant.contracts import (
    PriceHistory, ExposureBook, RiskAppetite, BacktestResult,
)


def walk_forward_validate(history: PriceHistory,
                          exposure: ExposureBook,
                          risk: RiskAppetite,
                          train_size: int = 1000,
                          test_size: int = 21,
                          step_size: int = 63,
                          label: str = "baseline",
                          forward_price_fn=None) -> BacktestResult:
    """
    CONTRACT
      in : full price history, exposure, risk appetite, window sizes, a label
           naming the configuration under test (e.g. "GBM", "GARCH-t")
      out: BacktestResult comparing the system's policy against
           perfect-foresight / no-hedge / naive baselines, plus the headline
           captured_fraction.

    METHOD (Phase 1): walk forward through history; at each decision date give
    the agent ONLY past data (slice at the cutoff -> no look-ahead), let it
    choose a policy, then score that policy on the ACTUAL prices that followed.
    Repeat; aggregate.

    USAGE: run with label="GBM", then re-run with label="GARCH-t", and compare
    captured_fraction to quantify the upgrade.
    """
    from hedging_assistant.agent.orchestrator import HedgingAgent

    # Default forward price: last price in the training window
    if forward_price_fn is None:
        forward_price_fn = lambda train_prices: float(train_prices[-1])

    all_prices = history.prices
    all_dates = history.dates
    n = len(all_prices)

    # 1. Compute decision dates: start at train_size, step by step_size,
    #    stop when train_size + test_size > len(history)
    decision_indices = []
    t = train_size
    while t + test_size <= n:
        decision_indices.append(t)
        t += step_size

    if not decision_indices:
        raise ValueError(
            f"No decision windows: history length {n} is too short for "
            f"train_size={train_size} + test_size={test_size}."
        )

    agent = HedgingAgent(risk)
    vol = exposure.volumes  # shape (horizon,)
    horizon = exposure.horizon

    strategy_costs = []
    perfect_foresight_costs = []
    no_hedge_costs = []
    naive_costs = []
    decision_dates = []

    for t in decision_indices:
        # 2a. Information cutoff -- agent sees ONLY past data
        train_prices = all_prices[t - train_size: t]
        train_dates = all_dates[t - train_size: t]

        # 2b. Build PriceHistory from the slice
        train_hist = PriceHistory(
            dates=train_dates,
            prices=train_prices,
            symbol=history.symbol,
        )

        # 2c. Compute forward price
        forward_price = float(forward_price_fn(train_prices))

        # 2d. Run agent
        rec = agent.recommend(train_hist, exposure, forward_price)
        frac = rec.policy.hedge_fractions  # shape (horizon,)

        # 2e. Actual prices for the test window
        actual_prices = all_prices[t: t + test_size]

        # 2f. Strategy cost
        strat_cost = 0.0
        for p in range(horizon):
            spot = actual_prices[p % test_size]
            strat_cost += frac[p] * vol[p] * forward_price + (1 - frac[p]) * vol[p] * spot
        strategy_costs.append(strat_cost)

        # 2g. No-hedge cost: buy everything at spot
        no_hedge_cost = float(np.sum(vol * actual_prices[:horizon]))
        no_hedge_costs.append(no_hedge_cost)

        # 2h. Perfect-foresight cost
        mean_actual = float(np.mean(actual_prices))
        total_vol = float(np.sum(vol))
        if forward_price < mean_actual:
            # hedge 100% -- forward is cheaper
            pf_cost = forward_price * total_vol
        else:
            # hedge 0% -- spot is cheaper
            pf_cost = mean_actual * total_vol
        perfect_foresight_costs.append(pf_cost)

        # 2i. Naive cost: fixed 50% hedge always
        naive_cost = (
            0.5 * total_vol * forward_price
            + 0.5 * float(np.sum(vol * actual_prices[:horizon]))
        )
        naive_costs.append(naive_cost)

        decision_dates.append(all_dates[t])

    # 3. Convert to arrays
    strategy_costs = np.array(strategy_costs)
    perfect_foresight_costs = np.array(perfect_foresight_costs)
    no_hedge_costs = np.array(no_hedge_costs)
    naive_costs = np.array(naive_costs)
    decision_dates = np.array(decision_dates)

    # 4. Captured fraction
    mean_strat = float(np.mean(strategy_costs))
    mean_pf = float(np.mean(perfect_foresight_costs))
    mean_no_hedge = float(np.mean(no_hedge_costs))
    captured_fraction = float(np.clip(
        1.0 - (mean_strat - mean_pf) / (mean_no_hedge - mean_pf + 1e-9),
        0.0, 1.0,
    ))

    # 5. Print summary
    mean_naive = float(np.mean(naive_costs))
    print(f"\n=== Walk-Forward Backtest: {label} ===")
    print(f"  Windows          : {len(decision_indices)}")
    print(f"  Mean strategy    : ${mean_strat:,.0f}")
    print(f"  Mean no-hedge    : ${mean_no_hedge:,.0f}")
    print(f"  Mean perfect-fsg : ${mean_pf:,.0f}")
    print(f"  Mean naive-50pct : ${mean_naive:,.0f}")
    print(f"  Captured fraction: {captured_fraction:.1%}")

    # 6. Return BacktestResult
    return BacktestResult(
        dates=decision_dates,
        strategy_costs=strategy_costs,
        perfect_foresight_costs=perfect_foresight_costs,
        no_hedge_costs=no_hedge_costs,
        naive_costs=naive_costs,
        captured_fraction=captured_fraction,
        label=label,
    )
