"""
LangGraph plan-and-execute workflow for the hedging assistant.

Implements the same 5-step pipeline as orchestrator.py but as a
LangGraph StateGraph with explicit nodes:
  node_assess -> node_forecast -> node_explore -> node_arbitrate -> node_explain

All nodes are deterministic Python (no LLM calls). Phase 2 will add LLM-driven
ASSESS and ARBITRATE nodes plus an LLM-written EXPLAIN rationale.
"""

from __future__ import annotations
from typing import Any

from langgraph.graph import StateGraph, END

from hedging_assistant.contracts import (
    PriceHistory, ExposureBook, RiskAppetite,
    StrategyType, StrategyParams,
    CandidateRecord, Recommendation,
)
from hedging_assistant.engines.forecaster import forecast
from hedging_assistant.engines.strategy_library import (
    build_policy, generate_staggered_candidates,
)
from hedging_assistant.engines.cost_simulator import simulate_cost
from hedging_assistant.engines.scorer import score_policy, evaluate_candidates


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(dict):
    """
    Typed dictionary carrying the workflow state between nodes.

    Keys:
        history         : PriceHistory
        exposure        : ExposureBook
        risk            : RiskAppetite
        forward_price   : float
        candidates      : list[StrategyParams] | None
        forecast_obj    : PriceForecast | None
        records         : list[CandidateRecord] | None
        no_hedge_cost   : CostDistribution | None
        best            : CandidateRecord | None
        recommendation  : Recommendation | None
        trace_messages  : list[str]
    """


def _initial_state(
    history: PriceHistory,
    exposure: ExposureBook,
    risk: RiskAppetite,
    forward_price: float,
) -> AgentState:
    return AgentState(
        history=history,
        exposure=exposure,
        risk=risk,
        forward_price=forward_price,
        candidates=None,
        forecast_obj=None,
        records=None,
        no_hedge_cost=None,
        best=None,
        recommendation=None,
        trace_messages=[],
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_assess(state: AgentState) -> AgentState:
    """
    ASSESS: Generate candidate StrategyParams to evaluate.
    BASELINE: enumerates a staggered fraction grid.
    UPGRADE (Phase 2): LLM inspects market state + risk appetite to prune/extend.
    """
    risk: RiskAppetite = state["risk"]
    candidates = generate_staggered_candidates(
        max_hedge=risk.max_hedge,
        n_steps=11,
        cap=risk.max_hedge,
    )
    msg = f"[node_assess] Generated {len(candidates)} staggered candidates."
    state["candidates"] = candidates
    state["trace_messages"] = state["trace_messages"] + [msg]
    print(msg)
    return state


def node_forecast(state: AgentState) -> AgentState:
    """
    FORECAST: Run GBM-Sobol forecaster to produce a PriceForecast.
    """
    history: PriceHistory = state["history"]
    exposure: ExposureBook = state["exposure"]
    fc = forecast(history, exposure.horizon, seed=0)
    msg = (
        f"[node_forecast] Produced {fc.n_paths} paths x {fc.horizon} steps "
        f"({fc.model_name})."
    )
    state["forecast_obj"] = fc
    state["trace_messages"] = state["trace_messages"] + [msg]
    print(msg)
    return state


def node_explore(state: AgentState) -> AgentState:
    """
    EXPLORE: Evaluate all candidates using evaluate_candidates and also
    compute no-hedge cost for baseline comparison.
    """
    forecast_obj = state["forecast_obj"]
    exposure: ExposureBook = state["exposure"]
    risk: RiskAppetite = state["risk"]
    forward_price: float = state["forward_price"]

    # Compute no-hedge baseline cost
    no_hedge_params = StrategyParams(
        strategy_type=StrategyType.STAGGERED,
        base_fraction=0.0,
    )
    no_hedge_cost = simulate_cost(
        forecast_obj, exposure, no_hedge_params, forward_price,
    )

    # Evaluate all candidates; evaluate_candidates returns list[dict]
    # with keys: params, cost, score
    results = evaluate_candidates(
        forecast_obj=forecast_obj,
        exposure=exposure,
        risk=risk,
        forward_price=forward_price,
    )

    # Convert to CandidateRecord list
    records = [
        CandidateRecord(
            params=r["params"],
            score=r["score"],
            accepted=False,
        )
        for r in results
    ]

    msg = (
        f"[node_explore] Evaluated {len(records)} candidates. "
        f"No-hedge mean cost: ${no_hedge_cost.mean:,.0f}."
    )
    state["records"] = records
    state["no_hedge_cost"] = no_hedge_cost
    state["trace_messages"] = state["trace_messages"] + [msg]
    print(msg)
    return state


def node_arbitrate(state: AgentState) -> AgentState:
    """
    ARBITRATE: Pick the best candidate (minimum blended score).
    BASELINE: deterministic argmin.
    UPGRADE (Phase 2): LLM reasons about trade-offs vs risk appetite.
    """
    records: list[CandidateRecord] = state["records"]
    best = min(records, key=lambda r: r.score.blended)
    best.accepted = True
    msg = (
        f"[node_arbitrate] Best candidate: "
        f"{best.params.base_fraction:.0%} staggered, "
        f"blended score={best.score.blended:,.0f}."
    )
    state["best"] = best
    state["trace_messages"] = state["trace_messages"] + [msg]
    print(msg)
    return state


def node_explain(state: AgentState) -> AgentState:
    """
    EXPLAIN: Build HedgingPolicy, write rationale, create Recommendation.
    BASELINE: template-based rationale string.
    UPGRADE (Phase 2): LLM writes nuanced explanation.
    """
    best: CandidateRecord = state["best"]
    forecast_obj = state["forecast_obj"]
    exposure: ExposureBook = state["exposure"]
    forward_price: float = state["forward_price"]
    records: list[CandidateRecord] = state["records"]

    policy = build_policy(best.params, forecast_obj)
    cost = simulate_cost(forecast_obj, exposure, best.params, forward_price)

    rationale = (
        f"Recommend hedging {best.params.base_fraction:.0%} of exposure "
        f"({best.params.strategy_type.value} strategy, cap={best.params.cap:.0%}). "
        f"Expected cost ${cost.mean:,.0f}; tail risk CVaR ${cost.cvar:,.0f} "
        f"(at 95th percentile). "
        f"Blended score {best.score.blended:,.0f} (lower is better)."
    )

    recommendation = Recommendation(
        policy=policy,
        cost=cost,
        score=best.score,
        rationale=rationale,
        trace=records,
        assumptions={
            "forward_price": forward_price,
            "model": forecast_obj.model_name,
        },
    )

    msg = "[node_explain] Recommendation built."
    state["recommendation"] = recommendation
    state["trace_messages"] = state["trace_messages"] + [msg]
    print(msg)
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    """Build and compile the LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    builder.add_node("assess", node_assess)
    builder.add_node("forecast", node_forecast)
    builder.add_node("explore", node_explore)
    builder.add_node("arbitrate", node_arbitrate)
    builder.add_node("explain", node_explain)

    builder.set_entry_point("assess")
    builder.add_edge("assess", "forecast")
    builder.add_edge("forecast", "explore")
    builder.add_edge("explore", "arbitrate")
    builder.add_edge("arbitrate", "explain")
    builder.add_edge("explain", END)

    return builder.compile()


# Compile once at module import
_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(
    history: PriceHistory,
    exposure: ExposureBook,
    risk: RiskAppetite,
    forward_price: float,
) -> Recommendation:
    """
    Run the full LangGraph plan-and-execute workflow and return a Recommendation.

    Args:
        history:       historical price data
        exposure:      volumes required per future period
        risk:          client risk appetite (weights, CVaR alpha, max hedge)
        forward_price: current forward/futures price for hedged volumes

    Returns:
        Recommendation with policy, cost distribution, score, rationale, trace
    """
    initial = _initial_state(history, exposure, risk, forward_price)
    final_state = _graph.invoke(initial)
    return final_state["recommendation"]
