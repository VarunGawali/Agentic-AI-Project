"""
Plan-and-execute orchestrator -- SKELETON.

The agent's 5 steps from the architecture diagram are explicit methods. In
Phase 1 these are plain Python (a simple loop over candidates). In Phase 2 the
same structure is wrapped in LangGraph and hosted on Foundry, and the LLM takes
over the ASSESS and ARBITRATE judgement steps and writes the EXPLAIN text.

The point of stubbing it this way: the agentic *structure* (assess -> forecast
-> explore -> arbitrate -> explain) exists from day one, and swapping the plain
logic for LLM-driven logic later does not change the pipeline shape.
"""

from __future__ import annotations
import numpy as np

from hedging_assistant.contracts import (
    PriceHistory, ExposureBook, RiskAppetite,
    StrategyType, StrategyParams, Recommendation, CandidateRecord,
)
from hedging_assistant.engines.forecaster import forecast
from hedging_assistant.engines.strategy_library import build_policy, generate_staggered_candidates
from hedging_assistant.engines.cost_simulator import simulate_cost
from hedging_assistant.engines.scorer import score_policy


class HedgingAgent:
    def __init__(self, risk: RiskAppetite):
        self.risk = risk

    # --- STEP 1 --------------------------------------------------------------
    def assess(self, history: PriceHistory, horizon: int) -> list[StrategyParams]:
        """
        Decide WHICH strategies/parameters to evaluate.
        BASELINE: enumerate staggered fractions.
        UPGRADE : LLM inspects market state + risk appetite to prune the space.
        """
        return generate_staggered_candidates(
            max_hedge=self.risk.max_hedge, n_steps=11, cap=self.risk.max_hedge
        )

    # --- STEP 2 --------------------------------------------------------------
    def forecast(self, history: PriceHistory, horizon: int):
        return forecast(history, horizon, seed=0)

    # --- STEP 3 --------------------------------------------------------------
    def explore(self, candidates, forecast_obj, exposure, forward_price):
        """Evaluate + score each candidate. Returns list[CandidateRecord]."""
        no_hedge = simulate_cost(
            forecast_obj, exposure,
            StrategyParams(StrategyType.STAGGERED, base_fraction=0.0),
            forward_price,
        )
        records = []
        for params in candidates:
            cost = simulate_cost(forecast_obj, exposure, params, forward_price)
            score = score_policy(cost, no_hedge, params, self.risk)
            records.append(CandidateRecord(params=params, score=score,
                                           accepted=False))
        return records, no_hedge

    # --- STEP 4 --------------------------------------------------------------
    def arbitrate(self, records: list[CandidateRecord]) -> CandidateRecord:
        """
        Pick the winning candidate.
        BASELINE: lowest blended score.
        UPGRADE : LLM reasons about trade-offs vs risk appetite.
        """
        best = min(records, key=lambda r: r.score.blended)
        best.accepted = True
        return best

    # --- STEP 5 --------------------------------------------------------------
    def explain(self, best: CandidateRecord, forecast_obj, exposure,
                forward_price, records) -> Recommendation:
        """Compose the recommendation + rationale + trace."""
        policy = build_policy(best.params, forecast_obj)
        cost = simulate_cost(forecast_obj, exposure, best.params, forward_price)
        rationale = (
            f"Recommend hedging {best.params.base_fraction:.0%} "
            f"({best.params.strategy_type.value}). Expected cost "
            f"${cost.mean:,.0f}; worst-case (CVaR) ${cost.cvar:,.0f}."
        )
        return Recommendation(
            policy=policy, cost=cost, score=best.score,
            rationale=rationale, trace=records,
            assumptions={"forward_price": forward_price,
                         "model": forecast_obj.model_name},
        )

    # --- FULL PASS -----------------------------------------------------------
    def recommend(self, history: PriceHistory, exposure: ExposureBook,
                  forward_price: float) -> Recommendation:
        horizon = exposure.horizon
        candidates = self.assess(history, horizon)
        fc = self.forecast(history, horizon)
        records, _ = self.explore(candidates, fc, exposure, forward_price)
        best = self.arbitrate(records)
        return self.explain(best, fc, exposure, forward_price, records)
