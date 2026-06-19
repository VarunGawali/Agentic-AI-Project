"""
Walk-forward backtest runner.

Loads synthetic history, runs the backtester with default parameters, and
prints side-by-side comparisons of the agent strategy vs baselines.

Usage:
    python run_backtest.py
"""

from __future__ import annotations
import numpy as np

from hedging_assistant.data.loader import (
    load_price_history, make_exposure_book, default_risk_appetite,
)
from hedging_assistant.backtest.validate import walk_forward_validate


def main():
    print("=" * 64)
    print(" HEDGING ASSISTANT -- WALK-FORWARD BACKTEST")
    print("=" * 64)

    # Load synthetic history (no EIA key needed)
    history = load_price_history(symbol="WTI")
    exposure = make_exposure_book(barrels_per_period=100_000, horizon=6)
    risk = default_risk_appetite()

    print(f"\nHistory : {len(history)} prices, {history.symbol}")
    print(f"Exposure: {exposure.volumes[0]:,.0f} bbl/month x {exposure.horizon} months")

    # --- Main backtest: agent with GBM-Sobol label ---
    result = walk_forward_validate(
        history=history,
        exposure=exposure,
        risk=risk,
        label="GBM-Sobol",
    )

    # --- Naive comparison: compute naive captured fraction from the same result ---
    mean_naive = float(np.mean(result.naive_costs))
    mean_pf = float(np.mean(result.perfect_foresight_costs))
    mean_no_hedge = float(np.mean(result.no_hedge_costs))
    naive_captured = float(np.clip(
        1.0 - (mean_naive - mean_pf) / (mean_no_hedge - mean_pf + 1e-9),
        0.0, 1.0,
    ))

    print("\n" + "=" * 64)
    print(" SIDE-BY-SIDE COMPARISON")
    print("=" * 64)
    print(f"  {'Strategy':<30} {'Mean Cost':>14}  {'Captured':>10}")
    print(f"  {'-'*30} {'-'*14}  {'-'*10}")
    print(f"  {'Perfect foresight (bound)':<30} ${mean_pf:>13,.0f}  {'100.0%':>10}")
    print(f"  {'Agent ({})'.format(result.label):<30} ${np.mean(result.strategy_costs):>13,.0f}  {result.captured_fraction:>10.1%}")
    print(f"  {'Naive 50% hedge':<30} ${mean_naive:>13,.0f}  {naive_captured:>10.1%}")
    print(f"  {'No hedge (spot)':<30} ${mean_no_hedge:>13,.0f}  {'0.0%':>10}")
    print()
    print(f"Agent captured_fraction  : {result.captured_fraction:.1%}")
    print(f"Naive-50pct captured_frac: {naive_captured:.1%}")
    print(f"Agent advantage over naive: {result.captured_fraction - naive_captured:+.1%}")
    print()
    print("Backtest complete.")


if __name__ == "__main__":
    main()
