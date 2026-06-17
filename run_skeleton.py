"""
End-to-end skeleton run.

Proves the pipeline's CONTRACTS fit together: data -> agent (forecast ->
explore -> arbitrate -> explain) -> recommendation. Uses the Phase-1 baseline
logic (GBM + staggered + plain MC + 2-factor). Run:  python -m hedging_assistant.run_skeleton
"""

from __future__ import annotations
import numpy as np

from hedging_assistant.data.loader import (
    load_price_history, make_exposure_book, default_risk_appetite,
)
from hedging_assistant.agent.orchestrator import HedgingAgent


def main():
    print("=" * 64)
    print(" HEDGING ASSISTANT -- SKELETON END-TO-END RUN")
    print("=" * 64)

    # --- Data layer ---
    history = load_price_history(symbol="WTI")
    exposure = make_exposure_book(barrels_per_period=100_000, horizon=6)
    risk = default_risk_appetite()
    forward_price = float(history.prices[-1])   # simple v1: forward = last spot

    print(f"\nData    : {len(history)} prices, {history.symbol}, "
          f"last ${forward_price:.1f}/bbl")
    print(f"Exposure: {exposure.volumes[0]:,.0f} bbl/{exposure.period_label} "
          f"x {exposure.horizon}")

    # --- Agent (5-step workflow) ---
    agent = HedgingAgent(risk)
    rec = agent.recommend(history, exposure, forward_price)

    # --- Output ---
    print("\n--- RECOMMENDATION " + "-" * 45)
    print(rec.rationale)
    print(f"\nPolicy schedule (hedge fraction/period): "
          f"{np.round(rec.policy.hedge_fractions, 2)}")
    print(f"Cost  P10/P50/P90: ${rec.cost.p10:,.0f} / "
          f"${rec.cost.p50:,.0f} / ${rec.cost.p90:,.0f}")
    print(f"Model : {rec.assumptions['model']}")

    print("\n--- DECISION TRACE (candidates considered) " + "-" * 20)
    for r in rec.trace:
        mark = " <== chosen" if r.accepted else ""
        print(f"  hedge {r.params.base_fraction:>4.0%}  "
              f"blended={r.score.blended:>14,.0f}{mark}")

    print("\nSkeleton ran end-to-end. Contracts fit. Ready for Phase 1 logic.")


if __name__ == "__main__":
    main()
