"""
LangGraph plan-and-execute workflow for the hedging assistant.

CHANGES:
  - Fixed AgentState to use TypedDict (required by LangGraph >= 1.x).
  - Phase 3: ASSESS, ARBITRATE, EXPLAIN nodes use Azure OpenAI GPT-4o when
    AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT are set.
    Falls back to deterministic Python silently when env vars are absent.
  - GPT-4o chosen for Azure stack: best reasoning/latency tradeoff for
    short structured financial prompts; temperature=0 for determinism.

Environment variables (all optional):
  AZURE_OPENAI_API_KEY      Azure OpenAI key
  AZURE_OPENAI_ENDPOINT     e.g. https://<resource>.openai.azure.com/
  AZURE_OPENAI_DEPLOYMENT   deployment name (default: gpt-4o)
  AZURE_OPENAI_API_VERSION  API version (default: 2024-02-01)
"""
from __future__ import annotations
import logging
import os, json

logger = logging.getLogger(__name__)
from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from hedging_assistant.contracts import (
    PriceHistory, ExposureBook, RiskAppetite,
    StrategyType, StrategyParams, CandidateRecord, Recommendation,
)
from hedging_assistant.engines.forecaster import forecast
from hedging_assistant.engines.strategy_library import build_policy, generate_staggered_candidates
from hedging_assistant.engines.cost_simulator import simulate_cost
from hedging_assistant.engines.scorer import score_policy, evaluate_candidates

# ---------------------------------------------------------------------------
# LLM — lazy init, graceful fallback
# ---------------------------------------------------------------------------
def _get_llm():
    key, endpoint = os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("AZURE_OPENAI_ENDPOINT")
    if not key or not endpoint:
        return None
    try:
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            azure_endpoint=endpoint, api_key=key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.0, max_tokens=512,
        )
    except Exception as e:
        logger.warning("[langgraph_agent] LLM init failed (%s); deterministic fallback.", e)
        return None

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    history: Any; exposure: Any; risk: Any; forward_price: float
    candidates: Any; forecast_obj: Any; records: Any
    no_hedge_cost: Any; best: Any; recommendation: Any
    trace_messages: list; llm_rationale: str

def _initial_state(history, exposure, risk, forward_price) -> AgentState:
    return AgentState(
        history=history, exposure=exposure, risk=risk, forward_price=forward_price,
        candidates=None, forecast_obj=None, records=None, no_hedge_cost=None,
        best=None, recommendation=None, trace_messages=[], llm_rationale="",
    )

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def node_assess(state: AgentState) -> AgentState:
    """LLM prunes candidate space from market context; falls back to 11-step grid."""
    risk, history = state["risk"], state["history"]
    llm = _get_llm()
    candidates = None
    if llm:
        try:
            import numpy as np, pandas as pd
            prices = np.asarray(history.prices)
            recent_vol   = float(np.std(np.diff(np.log(prices[-21:]))) * np.sqrt(252))
            recent_trend = float((prices[-1] / prices[-20] - 1) * 100)
            try:
                from hedging_assistant.engines.regime import detect_regime
                regime_label = detect_regime(pd.Series(prices)).label
            except Exception:
                regime_label = "unknown"
            prompt = (
                f"You are a crude oil procurement risk advisor.\n"
                f"Market context:\n"
                f"- Annualised vol (21-day): {recent_vol:.1%}\n"
                f"- 20-day price trend: {recent_trend:+.1f}%\n"
                f"- HMM regime: {regime_label}\n"
                f"- Client max hedge: {risk.max_hedge:.0%}, CVaR weight: {risk.w_cvar:.1f}\n\n"
                f"Return a JSON array of 7-11 hedge fractions (0.0–{risk.max_hedge:.1f}) to evaluate.\n"
                f"Bias toward higher fractions in high-vol/bullish regimes.\n"
                f"Always include 0.0 and {risk.max_hedge:.1f}.\n"
                f"Respond ONLY with a valid JSON array."
            )
            raw = llm.invoke(prompt).content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            fracs = sorted({float(f) for f in json.loads(raw) if 0.0 <= float(f) <= risk.max_hedge})
            if 0.0 not in fracs:
                fracs = [0.0] + fracs
            candidates = [StrategyParams(strategy_type=StrategyType.STAGGERED, base_fraction=f) for f in fracs]
            msg = f"[node_assess/LLM] {len(candidates)} candidates: {[f'{c.base_fraction:.0%}' for c in candidates]}"
        except Exception as e:
            logger.warning("[node_assess] LLM failed (%s); fallback.", e)
            candidates = None
    if candidates is None:
        candidates = generate_staggered_candidates(max_hedge=risk.max_hedge, n_steps=11, cap=risk.max_hedge)
        msg = f"[node_assess/det] {len(candidates)} staggered candidates."
    logger.info(msg)
    return {"candidates": candidates, "trace_messages": state.get("trace_messages", []) + [msg]}


def node_forecast(state: AgentState) -> AgentState:
    fc = forecast(state["history"], state["exposure"].horizon, seed=0)
    msg = f"[node_forecast] {fc.n_paths} paths × {fc.horizon} steps ({fc.model_name})."
    logger.info(msg)
    return {"forecast_obj": fc, "trace_messages": state.get("trace_messages", []) + [msg]}


def node_explore(state: AgentState) -> AgentState:
    fc, exposure, risk, fwd = state["forecast_obj"], state["exposure"], state["risk"], state["forward_price"]
    no_hedge_cost = simulate_cost(fc, exposure, StrategyParams(StrategyType.STAGGERED, base_fraction=0.0), fwd)
    results = evaluate_candidates(forecast_obj=fc, exposure=exposure, risk=risk, forward_price=fwd)
    records = [CandidateRecord(params=r["params"], score=r["score"], accepted=False) for r in results]
    msg = f"[node_explore] {len(records)} candidates. No-hedge: ${no_hedge_cost.mean:,.0f}."
    logger.info(msg)
    return {"records": records, "no_hedge_cost": no_hedge_cost, "trace_messages": state.get("trace_messages", []) + [msg]}


def node_arbitrate(state: AgentState) -> AgentState:
    """LLM reasons about tradeoffs and picks a winner; falls back to min(blended)."""
    records, risk = state["records"], state["risk"]
    llm = _get_llm()
    best = None
    if llm:
        try:
            rows = ["idx | hedge% | E[cost]$M | CVaR$M | blended"] + [
                f"{i:3d} | {r.params.base_fraction:6.0%} | {r.score.cost/1e6:9.2f} | {r.score.cvar/1e6:7.2f} | {r.score.blended/1e6:7.2f}"
                for i, r in enumerate(records)
            ]
            prompt = (
                f"You are a crude oil procurement risk advisor.\n\n"
                f"Strategies (lower blended = better):\n{chr(10).join(rows)}\n\n"
                f"Client: CVaR weight={risk.w_cvar:.1f}, max hedge={risk.max_hedge:.0%}, "
                f"opp-cost weight={risk.w_opportunity:.1f}, exec-risk weight={risk.w_execution:.1f}\n\n"
                f"If CVaR weight > 1.5, favour lower CVaR over cost minimisation.\n"
                f"Respond EXACTLY:\nREASON: <one sentence>\nINDEX: <integer>"
            )
            content = llm.invoke(prompt).content.strip()
            idx_line    = next((l for l in content.splitlines() if l.startswith("INDEX:")),  None)
            reason_line = next((l for l in content.splitlines() if l.startswith("REASON:")), None)
            if idx_line:
                idx  = max(0, min(int(idx_line.replace("INDEX:", "").strip()), len(records)-1))
                best = records[idx]
                reason = reason_line.replace("REASON:", "").strip() if reason_line else ""
                msg = f"[node_arbitrate/LLM] idx={idx} ({best.params.base_fraction:.0%}). {reason}"
        except Exception as e:
            logger.warning("[node_arbitrate] LLM failed (%s); fallback.", e)
            best = None
    if best is None:
        best = min(records, key=lambda r: r.score.blended)
        msg = f"[node_arbitrate/det] {best.params.base_fraction:.0%}, blended={best.score.blended:,.0f}."
    best.accepted = True
    logger.info(msg)
    return {"best": best, "trace_messages": state.get("trace_messages", []) + [msg]}


def node_explain(state: AgentState) -> AgentState:
    """LLM writes a plain-English rationale for a non-quant executive; falls back to template."""
    best, fc, exposure, fwd = state["best"], state["forecast_obj"], state["exposure"], state["forward_price"]
    records, risk = state["records"], state["risk"]
    policy = build_policy(best.params, fc)
    cost   = simulate_cost(fc, exposure, best.params, fwd)
    llm    = _get_llm()
    rationale = ""
    if llm:
        try:
            sorted_rec = sorted(records, key=lambda r: r.score.blended)
            ru = sorted_rec[1] if len(sorted_rec) > 1 else None
            ru_text = (f"Runner-up: {ru.params.base_fraction:.0%} hedge (blended {ru.score.blended:,.0f})." if ru else "")
            prompt = (
                f"Write a 2-3 sentence recommendation for a crude oil procurement executive.\n\n"
                f"Chosen: {best.params.base_fraction:.0%} staggered hedge\n"
                f"  Expected cost: ${cost.mean/1e6:.2f}M | CVaR 95th: ${cost.cvar/1e6:.2f}M\n"
                f"  Forward price locked: ${fwd:.1f}/bbl\n"
                f"  Schedule: {[round(float(f),2) for f in policy.hedge_fractions]}\n"
                f"{ru_text}\n\n"
                f"Rules: plain English, no bullets, no markdown. "
                f"Mention hedge %, expected cost, CVaR. End with a clear action statement. "
                f"Output only the recommendation text."
            )
            rationale = llm.invoke(prompt).content.strip()
            msg = "[node_explain/LLM] LLM-written rationale."
        except Exception as e:
            logger.warning("[node_explain] LLM failed (%s); fallback.", e)
            rationale = ""
    if not rationale:
        rationale = (
            f"Recommend hedging {best.params.base_fraction:.0%} of exposure "
            f"({best.params.strategy_type.value} strategy). "
            f"Expected cost ${cost.mean:,.0f}; CVaR ${cost.cvar:,.0f} (95th pct). "
            f"Blended score {best.score.blended:,.0f}."
        )
        msg = "[node_explain/det] Template rationale."
    rec = Recommendation(
        policy=policy, cost=cost, score=best.score, rationale=rationale, trace=records,
        assumptions={"forward_price": fwd, "model": fc.model_name},
    )
    logger.info(msg)
    return {"recommendation": rec, "llm_rationale": rationale, "trace_messages": state.get("trace_messages", []) + [msg]}

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def _build_graph() -> Any:
    b = StateGraph(AgentState)
    for name, fn in [("assess", node_assess), ("forecast", node_forecast),
                     ("explore", node_explore), ("arbitrate", node_arbitrate),
                     ("explain", node_explain)]:
        b.add_node(name, fn)
    b.set_entry_point("assess")
    b.add_edge("assess", "forecast"); b.add_edge("forecast", "explore")
    b.add_edge("explore", "arbitrate"); b.add_edge("arbitrate", "explain")
    b.add_edge("explain", END)
    return b.compile()

_graph = _build_graph()

def run_agent(history: PriceHistory, exposure: ExposureBook,
              risk: RiskAppetite, forward_price: float) -> Recommendation:
    """Run the full LangGraph workflow and return a Recommendation."""
    return _graph.invoke(_initial_state(history, exposure, risk, forward_price))["recommendation"]
