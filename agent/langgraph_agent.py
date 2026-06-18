"""
LangGraph plan-and-execute agentic workflow.

Phase 1: all 5 nodes are deterministic Python — no LLM calls.
Phase 2: ASSESS and ARBITRATE nodes will be replaced with LLM-driven logic
         (Claude via Azure AI Foundry). EXPLAIN will use an LLM to write the
         rationale. The graph structure stays identical.

Graph: assess -> forecast -> explore -> arbitrate -> explain -> END
"""

from __future__ import annotations
import operator
from typing import Any, Annotated

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from hedging_assistant.contracts import (
    PriceHistory, ExposureBook, RiskAppetite,
    StrategyType, StrategyParams, CandidateRecord, Recommendation,
)
from hedging_assistant.engines.forecaster import forecast
from hedging_assistant.engines.strategy_library import (
    generate_staggered_candidates, build_policy,
)
from hedging_assistant.engines.cost_simulator import simulate_cost
from hedging_assistant.engines.scorer import score_policy, evaluate_candidates


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # --- inputs (set once at invocation) ---
    history: Any            # PriceHistory
    exposure: Any           # ExposureBook
    risk: Any               # RiskAppetite
    forward_price: float
    # --- intermediate (filled by nodes) ---
    candidates: list        # list[StrategyParams]
    forecast_obj: Any       # PriceForecast
    candidate_records: list # list[CandidateRecord]
    no_hedge_cost: Any      # CostDistribution
    best_record: Any        # CandidateRecord
    recommendation: Any     # Recommendation
    # --- trace ---
    messages: Annotated[list, operator.add]  # append-only log
    step: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_assess(state: AgentState) -> dict:
    """
    STEP 1 — ASSESS: decide which strategies/params to evaluate.
    BASELINE: enumerate staggered fractions across the full hedge grid.
    PHASE 2:  LLM inspects market state + risk appetite to prune the space.
    """
    risk = state["risk"]
    candidates = generate_staggered_candidates(
        max_hedge=risk.max_hedge, n_steps=11, cap=risk.max_hedge
    )
    return {
        "candidates": candidates,
        "step": "ASSESS",
        "messages": [f"ASSESS: generated {len(candidates)} staggered candidates "
                     f"(0% → {risk.max_hedge:.0%}, step 10%)"],
    }


def node_forecast(state: AgentState) -> dict:
    """
    STEP 2 — FORECAST: generate price paths (shared across all candidates).
    Paths are generated ONCE here and reused in EXPLORE — never re-forecast in a loop.
    """
    history = state["history"]
    exposure = state["exposure"]
    fc = forecast(history, horizon=exposure.horizon, seed=42, use_cache=True)
    return {
        "forecast_obj": fc,
        "step": "FORECAST",
        "messages": [f"FORECAST: {fc.model_name} | {fc.n_paths:,} paths × "
                     f"{fc.horizon} periods | freq={fc.frequency}"],
    }


def node_explore(state: AgentState) -> dict:
    """
    STEP 3 — EXPLORE: simulate cost and score every candidate.
    Uses multiprocessing sweep from evaluate_candidates.
    """
    fc = state["forecast_obj"]
    exposure = state["exposure"]
    risk = state["risk"]
    fwd = state["forward_price"]

    # evaluate_candidates returns list of {params, cost, score}, best-first
    results = evaluate_candidates(fc, exposure, risk, fwd)

    # no-hedge baseline (0% hedge)
    no_hedge_params = StrategyParams(
        strategy_type=StrategyType.STAGGERED, base_fraction=0.0
    )
    no_hedge_cost = simulate_cost(fc, exposure, no_hedge_params, fwd,
                                  cvar_alpha=risk.cvar_alpha, mode="optimized")

    records = [
        CandidateRecord(params=r["params"], score=r["score"], accepted=False)
        for r in results
    ]
    return {
        "candidate_records": records,
        "no_hedge_cost": no_hedge_cost,
        "step": "EXPLORE",
        "messages": [f"EXPLORE: evaluated {len(records)} candidates | "
                     f"no-hedge mean=${no_hedge_cost.mean:,.0f} | "
                     f"no-hedge CVaR=${no_hedge_cost.cvar:,.0f}"],
    }


def node_arbitrate(state: AgentState) -> dict:
    """
    STEP 4 — ARBITRATE: pick the winning policy.
    BASELINE: lowest blended score wins.
    PHASE 2:  LLM reasons about trade-offs vs risk appetite + market context.
    """
    records = state["candidate_records"]
    best = min(records, key=lambda r: r.score.blended)
    best.accepted = True
    return {
        "best_record": best,
        "step": "ARBITRATE",
        "messages": [f"ARBITRATE: chose hedge={best.params.base_fraction:.0%} | "
                     f"blended={best.score.blended:,.0f} | "
                     f"cost=${best.score.cost:,.0f} | CVaR=${best.score.cvar:,.0f}"],
    }


def node_explain(state: AgentState) -> dict:
    """
    STEP 5 — EXPLAIN: compose the final Recommendation with rationale + trace.
    PHASE 2: LLM writes the rationale in plain English tailored to the audience.
    """
    best = state["best_record"]
    fc = state["forecast_obj"]
    exposure = state["exposure"]
    fwd = state["forward_price"]
    risk = state["risk"]
    records = state["candidate_records"]

    policy = build_policy(best.params, fc)
    cost = simulate_cost(fc, exposure, best.params, fwd,
                         cvar_alpha=risk.cvar_alpha, mode="optimized")

    ci_mean_lo, ci_mean_hi = cost.ci_mean
    ci_cvar_lo, ci_cvar_hi = cost.ci_cvar

    rationale = (
        f"Recommended policy: hedge {best.params.base_fraction:.0%} of exposure "
        f"using a staggered forward strategy.\n"
        f"  Expected cost : ${cost.mean:,.0f}  "
        f"[95% CI: ${ci_mean_lo:,.0f} – ${ci_mean_hi:,.0f}]\n"
        f"  CVaR (worst 5%): ${cost.cvar:,.0f}  "
        f"[95% CI: ${ci_cvar_lo:,.0f} – ${ci_cvar_hi:,.0f}]\n"
        f"  Cost range (P10–P90): ${cost.p10:,.0f} – ${cost.p90:,.0f}\n"
        f"  Forecast model: {fc.model_name} | {fc.n_paths:,} paths\n"
        f"  Candidates evaluated: {len(records)}"
    )

    rec = Recommendation(
        policy=policy,
        cost=cost,
        score=best.score,
        rationale=rationale,
        trace=records,
        assumptions={
            "forward_price": fwd,
            "model": fc.model_name,
            "n_paths": fc.n_paths,
            "frequency": fc.frequency,
        },
    )
    return {
        "recommendation": rec,
        "step": "EXPLAIN",
        "messages": [f"EXPLAIN: recommendation composed | "
                     f"rationale length={len(rationale)} chars"],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_hedging_graph():
    """Compile the 5-node LangGraph plan-and-execute graph."""
    graph = StateGraph(AgentState)

    graph.add_node("assess",    node_assess)
    graph.add_node("forecast",  node_forecast)
    graph.add_node("explore",   node_explore)
    graph.add_node("arbitrate", node_arbitrate)
    graph.add_node("explain",   node_explain)

    graph.set_entry_point("assess")
    graph.add_edge("assess",    "forecast")
    graph.add_edge("forecast",  "explore")
    graph.add_edge("explore",   "arbitrate")
    graph.add_edge("arbitrate", "explain")
    graph.add_edge("explain",   END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(
    history: PriceHistory,
    exposure: ExposureBook,
    risk: RiskAppetite,
    forward_price: float,
) -> Recommendation:
    """Run the full 5-step LangGraph workflow and return the Recommendation."""
    app = build_hedging_graph()

    initial_state: AgentState = {
        "history": history,
        "exposure": exposure,
        "risk": risk,
        "forward_price": forward_price,
        "candidates": [],
        "forecast_obj": None,
        "candidate_records": [],
        "no_hedge_cost": None,
        "best_record": None,
        "recommendation": None,
        "messages": [],
        "step": "START",
    }

    final_state = app.invoke(initial_state)

    print("\n[LangGraph Agent Trace]")
    for msg in final_state["messages"]:
        print(f"  ► {msg}")

    return final_state["recommendation"]
