"""
LangGraph agentic workflow for the hedging assistant.

Graph topology (Phase 4):

    START
      ├──────────────────────┐
      ↓                      ↓
    [node_forecast]   [node_market_intel]   ← parallel; no dependency
      ↓                      ↓
      └──────────────────────┘
                 ↓
           [node_assess]        ← LLM prunes strategy families + narrows param search space
                 ↓
      [node_explore_dispatch]   ← fans out one Send per strategy family
        /    |    |    \\
    [eval] [eval] [eval] [eval] ← parallel family evaluation (LangGraph Send API)
        \\    |    |    /
      [node_explore_collect]    ← merges results, adds CVaR-LP, builds records
                 ↓
           [node_arbitrate]     ← LLM adjusts weights for current regime, re-scores, picks winner
                 ↓
           [node_explain]       ← LLM writes executive rationale
                 ↓
               END

LLM nodes: market_intel, assess, arbitrate, explain
    - Use Azure OpenAI when AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT are set.
    - Graceful deterministic fallback when env vars absent.

node_market_intel is a tool-calling agent: the LLM decides which analytical tools
to invoke (price momentum, volatility regime, price percentile, sentiment) rather
than following a hardcoded sequence.

Environment variables (all optional):
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_DEPLOYMENT
    AZURE_OPENAI_API_VERSION
"""

from __future__ import annotations

import dataclasses
import json
import logging
import operator
import os
import time
from typing import Annotated, Any

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from hedging_assistant.contracts import (
    PriceHistory,
    ExposureBook,
    RiskAppetite,
    StrategyType,
    StrategyParams,
    CandidateRecord,
    Recommendation,
)

from hedging_assistant.engines.forecaster import forecast
from hedging_assistant.engines.strategy_library import (
    build_policy,
    generate_batch_candidates,
    build_dp_table,
)
from hedging_assistant.engines.cost_simulator import simulate_cost
from hedging_assistant.engines.scorer import evaluate_candidates, blend_scores


# ---------------------------------------------------------------------------
# LLM — module-level singleton, initialised once
# ---------------------------------------------------------------------------

LLM_SYSTEM_INSTRUCTIONS = """
You are a financial decision-support assistant for a crude oil procurement and hedging dashboard.

You do not perform numerical forecasting, optimization, Monte Carlo simulation, scoring, or CVaR calculation yourself. Those are handled by deterministic Python engines.

Your role is limited to:
1. Calling analytical tools to gather market intelligence.
2. Assessing market context and suggesting which strategy families to evaluate and with what parameter ranges.
3. Reasoning over already-computed candidate strategy scores and adjusting risk weights for the current regime.
4. Writing concise, business-friendly explanations for procurement executives.

Important rules:
- Never invent prices, costs, hedge ratios, CVaR values, or candidate results.
- Use only the numbers provided in the prompt or returned by tool calls.
- If asked to choose a strategy, choose only from the provided candidate indices.
- For JSON requests, respond with valid JSON only — no markdown fences.
- For arbitration requests, respond exactly in the requested format.
- For executive explanations, use plain English, no markdown, no bullets.
"""

_LLM_CLIENT = None
_LLM_INIT_ATTEMPTED = False


def _get_llm():
    global _LLM_CLIENT, _LLM_INIT_ATTEMPTED
    if _LLM_INIT_ATTEMPTED:
        return _LLM_CLIENT
    _LLM_INIT_ATTEMPTED = True
    key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not key or not endpoint:
        return None
    try:
        from openai import AzureOpenAI
        _LLM_CLIENT = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    except Exception as exc:
        logger.warning("LLM init failed (%s); deterministic fallback active.", exc)
    return _LLM_CLIENT


def _llm_invoke(client, prompt: str, max_tokens: int = 512, max_retries: int = 3) -> str:
    from openai import RateLimitError, APIStatusError, APIConnectionError
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment,
                temperature=0.0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            wait = 2 ** attempt
            logger.warning("Rate limit; retrying in %ds.", wait)
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise
        except APIStatusError as exc:
            if exc.status_code and exc.status_code >= 500:
                wait = 2 ** attempt
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
            raise
        except APIConnectionError:
            wait = 2 ** attempt
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("LLM invocation failed after all retries.")


def _llm_invoke_with_tools(
    client,
    messages: list[dict],
    tools: list[dict],
    max_tokens: int = 512,
    max_tool_rounds: int = 4,
) -> tuple[str, list[dict]]:
    """
    Run the tool-calling loop for node_market_intel.

    Returns (final_text, updated_messages).
    The LLM decides which tools to call; we execute them and feed results back
    until the model emits a final text response (finish_reason != "tool_calls").
    """
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

    for _ in range(max_tool_rounds):
        response = client.chat.completions.create(
            model=deployment,
            temperature=0.0,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        choice = response.choices[0]
        messages.append({"role": "assistant", "content": choice.message.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in (choice.message.tool_calls or [])
        ]})

        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            return choice.message.content or "", messages

        for tc in choice.message.tool_calls:
            result = _execute_market_tool(tc.function.name, tc.function.arguments, messages)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    # Safety: ask for a final answer if we hit the round limit
    response = client.chat.completions.create(
        model=deployment, temperature=0.0, max_tokens=max_tokens, messages=messages,
    )
    return response.choices[0].message.content or "", messages


# ---------------------------------------------------------------------------
# Market intel tools — executed locally using price history from state
# These are injected into the tool context via closure over `_prices_ref`
# ---------------------------------------------------------------------------

_prices_ref: np.ndarray | None = None   # set at start of node_market_intel


def _execute_market_tool(name: str, arguments_json: str, _messages: list) -> str:
    prices = _prices_ref
    if prices is None or len(prices) < 2:
        return json.dumps({"error": "price history unavailable"})

    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        args = {}

    if name == "compute_price_momentum":
        n = int(args.get("lookback_days", 20))
        n = max(2, min(n, len(prices) - 1))
        pct = float((prices[-1] / prices[-n] - 1.0) * 100.0)
        return json.dumps({"lookback_days": n, "price_change_pct": round(pct, 2)})

    if name == "compute_volatility_regime":
        n = int(args.get("lookback_days", 30))
        n = max(5, min(n, len(prices) - 1))
        log_rets = np.diff(np.log(prices[-n:]))
        ann_vol = float(np.std(log_rets, ddof=1) * np.sqrt(252) * 100.0)
        regime = "high" if ann_vol > 35 else ("low" if ann_vol < 15 else "normal")
        return json.dumps({"lookback_days": n, "annualised_vol_pct": round(ann_vol, 2), "regime": regime})

    if name == "compute_price_percentile":
        n = int(args.get("lookback_days", 252))
        n = max(10, min(n, len(prices)))
        window = prices[-n:]
        pct = float(np.sum(window <= prices[-1]) / len(window) * 100.0)
        return json.dumps({"lookback_days": n, "price_percentile": round(pct, 1),
                           "current_price": round(float(prices[-1]), 2),
                           "range_low": round(float(window.min()), 2),
                           "range_high": round(float(window.max()), 2)})

    if name == "get_market_sentiment":
        # Derive sentiment from recent price action (web search not required)
        mom_20 = float((prices[-1] / prices[-min(20, len(prices)-1)] - 1.0) * 100.0)
        log_rets = np.diff(np.log(prices[-min(30, len(prices)):]))
        vol = float(np.std(log_rets, ddof=1) * np.sqrt(252) * 100.0)
        sentiment = "bullish" if mom_20 > 3 else ("bearish" if mom_20 < -3 else "neutral")
        return json.dumps({
            "sentiment": sentiment,
            "momentum_20d_pct": round(mom_20, 2),
            "vol_regime": "high" if vol > 35 else ("low" if vol < 15 else "normal"),
            "note": "derived from price action; no external news source"
        })

    return json.dumps({"error": f"unknown tool: {name}"})


_MARKET_INTEL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compute_price_momentum",
            "description": "Compute N-day price change % from recent price history to assess trend direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lookback_days": {"type": "integer", "description": "Lookback window (e.g. 10, 20, 60)"}
                },
                "required": ["lookback_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_volatility_regime",
            "description": "Compute annualised realised volatility over N trading days and classify regime (high/normal/low).",
            "parameters": {
                "type": "object",
                "properties": {
                    "lookback_days": {"type": "integer", "description": "Lookback window in trading days"}
                },
                "required": ["lookback_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_price_percentile",
            "description": "Return where the current price sits within its N-day range (0=at lows, 100=at highs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "lookback_days": {"type": "integer", "description": "Historical window (e.g. 252 for 1-year)"}
                },
                "required": ["lookback_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_sentiment",
            "description": "Returns overall crude oil market sentiment (bullish/neutral/bearish) based on price action analysis.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    # Inputs
    history: PriceHistory
    exposure: ExposureBook
    risk: RiskAppetite
    forward_price: Any
    frequency: str
    n_paths: int
    seed: int
    calibration_window: int | None
    distribution: str
    use_regime: bool
    model: str

    # Market intelligence (produced by node_market_intel, consumed by assess + arbitrate)
    market_context: dict

    # Strategy candidates and filtering
    candidates: list[StrategyParams]
    strategy_families: list[str]       # pruned family list from node_assess
    param_hints: dict                  # narrowed parameter ranges from node_assess

    # Fan-out result accumulator — operator.add merges lists from parallel evaluate_family nodes
    family_results: Annotated[list, operator.add]

    # Forecast
    forecast_obj: Any

    # Evaluation outputs
    results: list[dict]
    records: list[CandidateRecord]
    no_hedge_cost: Any

    # Arbitration
    best: dict
    adjusted_risk: RiskAppetite        # regime-adjusted weights from node_arbitrate

    # Final output
    recommendation: Recommendation
    trace_messages: list[str]
    llm_rationale: str


def _initial_state(
    history: PriceHistory,
    exposure: ExposureBook,
    risk: RiskAppetite,
    forward_price: Any,
    frequency: str = "M",
    n_paths: int = 4096,
    seed: int = 42,
    calibration_window: int | None = 1000,
    distribution: str = "normal",
    use_regime: bool = False,
    model: str = "xgb-garch-t",
) -> AgentState:
    return {
        "history": history,
        "exposure": exposure,
        "risk": risk,
        "forward_price": forward_price,
        "frequency": frequency,
        "n_paths": n_paths,
        "seed": seed,
        "calibration_window": calibration_window,
        "distribution": distribution,
        "use_regime": use_regime,
        "model": model,
        "market_context": {},
        "candidates": [],
        "strategy_families": [],
        "param_hints": {},
        "family_results": [],
        "forecast_obj": None,
        "results": [],
        "records": [],
        "no_hedge_cost": None,
        "best": {},
        "adjusted_risk": risk,
        "recommendation": None,
        "trace_messages": [],
        "llm_rationale": "",
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _selected_hedge_fraction(params: StrategyParams) -> float:
    if params.strategy_type == StrategyType.CVAR_LP and params.fixed_fractions is not None:
        return float(np.mean(params.fixed_fractions))
    return float(params.base_fraction)


def _strategy_display(params: StrategyParams) -> str:
    hedge_pct = _selected_hedge_fraction(params) * 100.0
    if params.strategy_type == StrategyType.CVAR_LP:
        return f"CVaR-LP optimized schedule, avg hedge {hedge_pct:.0f}%"
    return f"{hedge_pct:.0f}% {params.strategy_type.value}"


def _build_fan_data(paths: np.ndarray) -> dict:
    return {
        "p10": np.percentile(paths, 10, axis=0).tolist(),
        "p25": np.percentile(paths, 25, axis=0).tolist(),
        "p50": np.percentile(paths, 50, axis=0).tolist(),
        "p75": np.percentile(paths, 75, axis=0).tolist(),
        "p90": np.percentile(paths, 90, axis=0).tolist(),
    }


def _build_cost_histogram(strategy_costs, no_hedge_costs, cvar, bins=40):
    strategy_counts, strategy_edges = np.histogram(strategy_costs / 1e6, bins=bins)
    no_hedge_counts, no_hedge_edges = np.histogram(no_hedge_costs / 1e6, bins=bins)
    return {
        "strategy": {"counts": strategy_counts.tolist(), "edges": strategy_edges.tolist()},
        "no_hedge": {"counts": no_hedge_counts.tolist(), "edges": no_hedge_edges.tolist()},
        "cvar_line": round(cvar / 1e6, 2),
    }


# ---------------------------------------------------------------------------
# Node: market_intel  (parallel with node_forecast)
# ---------------------------------------------------------------------------

def node_market_intel(state: AgentState) -> dict:
    """
    Tool-calling LLM agent that gathers market intelligence.

    The LLM decides which analytical tools to call (price momentum,
    volatility regime, price percentile, sentiment) rather than following
    a hardcoded sequence. Results are stored in market_context and used
    by node_assess (strategy pruning + param narrowing) and node_arbitrate
    (regime-aware weight adjustment).

    Deterministic fallback: computes context from price history directly.
    """
    global _prices_ref

    history: PriceHistory = state["history"]
    prices = np.asarray(history.prices, dtype=float)
    _prices_ref = prices

    # --- Deterministic baseline (always computed; used if LLM unavailable) ---
    log_rets_30 = np.diff(np.log(prices[-min(31, len(prices)):]))
    ann_vol = float(np.std(log_rets_30, ddof=1) * np.sqrt(252) * 100.0)
    mom_20 = float((prices[-1] / prices[-min(21, len(prices)-1)] - 1.0) * 100.0)
    pct_252 = float(np.sum(prices[-min(252, len(prices)):] <= prices[-1]) / min(252, len(prices)) * 100.0)

    vol_regime = "high" if ann_vol > 35 else ("low" if ann_vol < 15 else "normal")
    price_trend = "bullish" if mom_20 > 3 else ("bearish" if mom_20 < -3 else "neutral")

    fallback_context = {
        "price_trend": price_trend,
        "vol_regime": vol_regime,
        "price_percentile_1y": round(pct_252, 1),
        "annualised_vol_pct": round(ann_vol, 2),
        "momentum_20d_pct": round(mom_20, 2),
        "key_risks": [],
        "source": "deterministic",
    }

    llm = _get_llm()
    if not llm:
        msg = f"[node_market_intel/det] trend={price_trend}, vol={vol_regime}, percentile={pct_252:.0f}%"
        logger.info("%s", msg)
        return {
            "market_context": fallback_context,
            "trace_messages": state.get("trace_messages", []) + [msg],
        }

    try:
        user_prompt = (
            "You are analysing the crude oil market to support a hedging decision.\n\n"
            f"Price history: {len(prices)} trading days available. "
            f"Latest price: ${prices[-1]:.2f}/bbl.\n\n"
            "Use the available tools to gather the market intelligence you need, "
            "then return a JSON object with these exact keys:\n"
            "  price_trend       (bullish|bearish|neutral)\n"
            "  vol_regime        (high|normal|low)\n"
            "  price_percentile_1y (0-100)\n"
            "  key_risks         (array of up to 3 short strings)\n"
            "  recommended_strategy_families (array from: staggered, trigger, volatility, hybrid)\n"
            "  trigger_thresholds (array of 2-3 floats between 0.90 and 1.20)\n"
            "  vol_scale_ks      (array of 2-3 floats between 0.0 and 3.0)\n"
            "  w_cvar_boost      (float, 0.0-0.5: extra CVaR weight for this regime; keep small)\n"
            "  w_opportunity_boost (float, 0.0-1.0: extra opportunity weight)\n\n"
            "Respond with ONLY valid JSON after calling the tools."
        )

        messages = [
            {"role": "system", "content": LLM_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ]

        final_text, _ = _llm_invoke_with_tools(
            client=llm,
            messages=messages,
            tools=_MARKET_INTEL_TOOLS,
            max_tokens=512,
        )

        # Strip markdown fences if present
        text = final_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1].replace("json", "").strip()

        ctx = json.loads(text)
        ctx["source"] = "llm"

        msg = (
            f"[node_market_intel/LLM] trend={ctx.get('price_trend')}, "
            f"vol={ctx.get('vol_regime')}, "
            f"families={ctx.get('recommended_strategy_families')}"
        )

    except Exception as exc:
        logger.warning("node_market_intel: LLM failed (%s); using deterministic context.", exc)
        ctx = fallback_context
        msg = f"[node_market_intel/det-fallback] {exc}"

    logger.info("%s", msg)
    return {
        "market_context": ctx,
        "trace_messages": state.get("trace_messages", []) + [msg],
    }


# ---------------------------------------------------------------------------
# Node: forecast  (parallel with node_market_intel)
# ---------------------------------------------------------------------------

def node_forecast(state: AgentState) -> dict:
    history: PriceHistory = state["history"]
    exposure: ExposureBook = state["exposure"]

    forecast_obj = forecast(
        history=history,
        horizon=exposure.horizon,
        frequency=state.get("frequency", "M"),
        n_paths=state.get("n_paths", 4096),
        seed=state.get("seed", 42),
        calibration_window=state.get("calibration_window", 1000),
        use_cache=True,
        distribution=state.get("distribution", "normal"),
        use_regime=state.get("use_regime", False),
        model=state.get("model", "xgb-garch-t"),
    )

    msg = (
        f"[node_forecast] {forecast_obj.n_paths} paths × "
        f"{forecast_obj.horizon} steps ({forecast_obj.model_name})."
    )
    logger.info("%s", msg)
    return {
        "forecast_obj": forecast_obj,
        "trace_messages": state.get("trace_messages", []) + [msg],
    }


# ---------------------------------------------------------------------------
# Node: assess  (runs after both parallel nodes complete)
# ---------------------------------------------------------------------------

def node_assess(state: AgentState) -> dict:
    """
    Prune strategy families and narrow parameter search space using market_context.

    The LLM uses the market intelligence produced by node_market_intel to:
    1. Select which strategy families to evaluate (fewer candidates → faster explore).
    2. Propose tighter parameter ranges for trigger thresholds and vol_scale_k.

    Deterministic fallback: use the full family set with default parameter ranges.
    """
    risk: RiskAppetite = state["risk"]
    market_ctx: dict = state.get("market_context", {})

    _ALL_FAMILIES = [
        StrategyType.STAGGERED,
        StrategyType.TRIGGER,
        StrategyType.VOLATILITY,
        StrategyType.HYBRID,
    ]

    _FAMILY_MAP = {
        "staggered": StrategyType.STAGGERED,
        "trigger": StrategyType.TRIGGER,
        "volatility": StrategyType.VOLATILITY,
        "hybrid": StrategyType.HYBRID,
    }

    strategy_types = _ALL_FAMILIES
    param_hints: dict = {}

    # --- Use market_context directly if it has LLM recommendations ---
    if market_ctx.get("recommended_strategy_families"):
        try:
            parsed = [
                _FAMILY_MAP[f.lower()]
                for f in market_ctx["recommended_strategy_families"]
                if isinstance(f, str) and f.lower() in _FAMILY_MAP
            ]
            if parsed:
                strategy_types = list(dict.fromkeys(parsed))
        except Exception:
            pass

    if market_ctx.get("trigger_thresholds"):
        param_hints["trigger_thresholds"] = market_ctx["trigger_thresholds"]
        param_hints["hybrid_trigger_thresholds"] = market_ctx["trigger_thresholds"]

    if market_ctx.get("vol_scale_ks"):
        param_hints["vol_scale_ks"] = market_ctx["vol_scale_ks"]
        param_hints["hybrid_vol_scale_ks"] = market_ctx["vol_scale_ks"]

    # --- If LLM did not provide families but is available, ask it ---
    if not market_ctx.get("recommended_strategy_families"):
        llm = _get_llm()
        if llm:
            try:
                history: PriceHistory = state["history"]
                prices = np.asarray(history.prices, dtype=float)
                recent_vol = float(np.std(np.diff(np.log(prices[-30:])), ddof=1) * np.sqrt(252))
                recent_trend = float((prices[-1] / prices[-20] - 1.0) * 100.0)

                prompt = (
                    "Crude oil market context:\n"
                    f"  Annualised vol: {recent_vol:.2%}\n"
                    f"  20-day trend: {recent_trend:+.2f}%\n"
                    f"  Vol regime: {market_ctx.get('vol_regime', 'unknown')}\n"
                    f"  Price trend: {market_ctx.get('price_trend', 'unknown')}\n"
                    f"  Client max hedge: {risk.max_hedge:.0%}\n"
                    f"  CVaR weight: {risk.w_cvar:.2f}\n\n"
                    "Return JSON with keys:\n"
                    "  families: array of strategy names (staggered/trigger/volatility/hybrid)\n"
                    "  trigger_thresholds: array of 2-3 floats\n"
                    "  vol_scale_ks: array of 2-3 floats\n"
                )

                raw = _llm_invoke(llm, prompt, max_tokens=256)
                if raw.startswith("```"):
                    raw = raw.split("```")[1].replace("json", "").strip()

                parsed_resp = json.loads(raw)

                families = [
                    _FAMILY_MAP[f.lower()]
                    for f in parsed_resp.get("families", [])
                    if isinstance(f, str) and f.lower() in _FAMILY_MAP
                ]
                if families:
                    strategy_types = list(dict.fromkeys(families))

                if parsed_resp.get("trigger_thresholds"):
                    param_hints["trigger_thresholds"] = parsed_resp["trigger_thresholds"]
                    param_hints["hybrid_trigger_thresholds"] = parsed_resp["trigger_thresholds"]
                if parsed_resp.get("vol_scale_ks"):
                    param_hints["vol_scale_ks"] = parsed_resp["vol_scale_ks"]
                    param_hints["hybrid_vol_scale_ks"] = parsed_resp["vol_scale_ks"]

            except Exception as exc:
                logger.warning("node_assess: LLM failed (%s); using full candidate set.", exc)

    family_names = [s.value for s in strategy_types]
    msg = (
        f"[node_assess] Families: {family_names}. "
        f"Param hints: {list(param_hints.keys()) or 'defaults'}."
    )
    logger.info("%s", msg)

    return {
        "strategy_families": family_names,
        "param_hints": param_hints,
        "trace_messages": state.get("trace_messages", []) + [msg],
    }


# ---------------------------------------------------------------------------
# Node: explore_dispatch  (fans out one Send per strategy family)
# ---------------------------------------------------------------------------

def node_explore_dispatch(state: AgentState) -> list[Send]:
    """
    Fan out: one evaluate_family node per strategy family.
    LangGraph runs them concurrently and merges family_results via operator.add.
    """
    risk: RiskAppetite = state["risk"]
    families_raw = state.get("strategy_families") or [
        StrategyType.STAGGERED.value,
        StrategyType.TRIGGER.value,
        StrategyType.VOLATILITY.value,
        StrategyType.HYBRID.value,
    ]
    param_hints = state.get("param_hints", {})

    _FAMILY_MAP = {
        "staggered": StrategyType.STAGGERED,
        "trigger": StrategyType.TRIGGER,
        "volatility": StrategyType.VOLATILITY,
        "hybrid": StrategyType.HYBRID,
    }

    sends = []
    for family_name in families_raw:
        stype = _FAMILY_MAP.get(family_name.lower())
        if stype is None:
            continue

        family_candidates = generate_batch_candidates(
            strategy_types=[stype],
            max_hedge=risk.max_hedge,
            n_steps=5,
            cap=risk.max_hedge,
            param_hints=param_hints,
        )

        sends.append(Send("evaluate_family", {
            **state,
            "current_family": family_name,
            "family_candidates": family_candidates,
            "family_results": [],   # reset accumulator for this Send
        }))

    return sends


# ---------------------------------------------------------------------------
# Node: evaluate_family  (one instance per family, run in parallel via Send)
# ---------------------------------------------------------------------------

def _extract_history_arrays(history: "PriceHistory | None") -> "tuple[np.ndarray | None, float | None]":
    """Extract monthly price array and long-run vol from PriceHistory for path-dependent strategies."""
    if history is None:
        return None, None
    prices = np.asarray(history.prices, dtype=float)
    if len(prices) < 2:
        return None, None
    long_run_vol = float(np.std(np.diff(np.log(prices[prices > 0])))) if len(prices) > 1 else None
    return prices, long_run_vol


def node_evaluate_family(state: AgentState) -> dict:
    """
    Evaluate all candidates for one strategy family.
    Results are merged into family_results via the operator.add reducer.
    """
    family_name: str = state["current_family"]
    candidates: list[StrategyParams] = state.get("family_candidates", [])
    forecast_obj = state["forecast_obj"]
    exposure: ExposureBook = state["exposure"]
    risk: RiskAppetite = state["risk"]
    forward_price = state["forward_price"]

    if not candidates:
        logger.info("[evaluate_family/%s] No candidates — skipped.", family_name)
        return {"family_results": []}

    price_history, long_run_vol = _extract_history_arrays(state.get("history"))

    results = evaluate_candidates(
        forecast_obj=forecast_obj,
        exposure=exposure,
        risk=risk,
        forward_price=forward_price,
        candidates=candidates,
        mode="accurate",
        compute_ci=False,
        price_history=price_history,
        long_run_vol=long_run_vol,
    )

    logger.info("[evaluate_family/%s] %d candidates evaluated.", family_name, len(results))
    return {"family_results": results}


# ---------------------------------------------------------------------------
# Node: explore_collect  (fan-in: merge family results, add CVaR-LP, build records)
# ---------------------------------------------------------------------------

def node_explore_collect(state: AgentState) -> dict:
    """
    Collect all family_results from the fan-out, add CVaR-LP candidate,
    compute no-hedge baseline, and build CandidateRecord list.
    """
    forecast_obj = state["forecast_obj"]
    exposure: ExposureBook = state["exposure"]
    risk: RiskAppetite = state["risk"]
    forward_price = state["forward_price"]

    all_results: list[dict] = list(state.get("family_results", []))

    price_history, long_run_vol = _extract_history_arrays(state.get("history"))

    # --- CVaR-LP optimized candidate ---
    lp_msg = "[node_explore_collect] CVaR-LP skipped."
    try:
        from hedging_assistant.engines.optimizer import optimize_cvar_lp_params

        cvar_lp_params = optimize_cvar_lp_params(
            forecast_obj=forecast_obj,
            exposure=exposure,
            forward_price=forward_price,
            cvar_alpha=risk.cvar_alpha,
            cost_weight=risk.w_cost,
            cvar_weight=risk.w_cvar,
            opportunity_weight=risk.w_opportunity,
            execution_weight=risk.w_execution,
            max_hedge=risk.max_hedge,
        )

        lp_results = evaluate_candidates(
            forecast_obj=forecast_obj,
            exposure=exposure,
            risk=risk,
            forward_price=forward_price,
            candidates=[cvar_lp_params],
            mode="accurate",
            compute_ci=False,
            price_history=price_history,
            long_run_vol=long_run_vol,
        )
        all_results.extend(lp_results)
        lp_msg = "[node_explore_collect] CVaR-LP candidate added."

    except Exception as exc:
        lp_msg = f"[node_explore_collect] CVaR-LP skipped: {exc}"

    logger.info("%s", lp_msg)

    # --- DP-optimal candidate (Bellman backward induction over price/vol state) ---
    dp_msg = "[node_explore_collect] DP-optimal skipped."
    try:
        dp_table = build_dp_table(
            forecast_obj=forecast_obj,
            exposure=exposure,
            forward_price=forward_price,
            max_hedge=risk.max_hedge,
            n_actions=11,
            cost_weight=risk.w_cost,
            cvar_weight=risk.w_cvar,
            cvar_alpha=risk.cvar_alpha,
            price_history=price_history,
            long_run_vol=long_run_vol,
        )

        if dp_table:
            dp_mean_frac = float(np.mean(list(dp_table.values())))
        else:
            dp_mean_frac = float(risk.max_hedge) * 0.5

        dp_params = StrategyParams(
            strategy_type=StrategyType.DP_OPTIMAL,
            base_fraction=min(dp_mean_frac, float(risk.max_hedge)),
            cap=risk.max_hedge,
            dp_table=dp_table,
        )

        dp_results = evaluate_candidates(
            forecast_obj=forecast_obj,
            exposure=exposure,
            risk=risk,
            forward_price=forward_price,
            candidates=[dp_params],
            mode="accurate",
            compute_ci=False,
            price_history=price_history,
            long_run_vol=long_run_vol,
        )
        all_results.extend(dp_results)
        dp_msg = (
            f"[node_explore_collect] DP-optimal candidate added "
            f"(mean hedge {dp_mean_frac:.0%}, {len(dp_table)} states)."
        )

    except Exception as exc:
        dp_msg = f"[node_explore_collect] DP-optimal skipped: {exc}"

    logger.info("%s", dp_msg)

    # --- No-hedge baseline ---
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

    # --- Pool-level normalize + blend, then sort (magnitude-invariant ranking) ---
    all_results = blend_scores(all_results, risk)

    records = []
    for item in all_results:
        params = item["params"]
        cost = item["cost"]
        score = item["score"]
        note = (
            f"Strategy={params.strategy_type.value}; "
            f"mean=${cost.mean:,.0f}; CVaR=${cost.cvar:,.0f}; "
            f"opp=${score.opportunity_cost:,.0f}; exec=${score.execution_risk:,.0f}; "
            f"blended_idx={score.blended:.3f}"
        )
        records.append(CandidateRecord(params=params, score=score, accepted=False, note=note))

    msg = (
        f"[node_explore_collect] {len(all_results)} total candidates. "
        f"No-hedge mean=${no_hedge_cost.mean:,.0f}."
    )
    logger.info("%s", msg)

    return {
        "results": all_results,
        "records": records,
        "no_hedge_cost": no_hedge_cost,
        "trace_messages": state.get("trace_messages", []) + [lp_msg, dp_msg, msg],
    }


# ---------------------------------------------------------------------------
# Node: arbitrate
# ---------------------------------------------------------------------------

def node_arbitrate(state: AgentState) -> dict:
    """
    Pick the best candidate.

    If LLM is available:
        1. Use market_context to propose regime-adjusted risk weights.
        2. Re-score candidates with adjusted weights.
        3. LLM picks the winner from the re-scored list with a reason.

    Deterministic fallback: min(blended_score) from original weights.
    """
    results: list[dict] = state["results"]
    records: list[CandidateRecord] = state["records"]
    risk: RiskAppetite = state["risk"]
    market_ctx: dict = state.get("market_context", {})

    if not results:
        raise ValueError("No candidate results available for arbitration.")

    llm = _get_llm()
    best = None
    adjusted_risk = risk

    if llm:
        try:
            # Step 1 — derive regime-adjusted weights from market_context
            w_cvar_boost = float(market_ctx.get("w_cvar_boost", 0.0))
            w_opp_boost = float(market_ctx.get("w_opportunity_boost", 0.0))

            # Cap boosts so LLM cannot override more than 50% of user weights
            w_cvar_boost = min(w_cvar_boost, risk.w_cvar * 0.5)
            w_opp_boost = min(w_opp_boost, risk.w_opportunity * 0.5)

            if w_cvar_boost != 0.0 or w_opp_boost != 0.0:
                adjusted_risk = dataclasses.replace(
                    risk,
                    w_cvar=max(0.0, risk.w_cvar + w_cvar_boost),
                    w_opportunity=max(0.0, risk.w_opportunity + w_opp_boost),
                )

                # Re-blend with adjusted weights on copied score objects so the
                # original (collected) ranking is not mutated. Raw factors are
                # weight-independent; only the normalization weights change.
                rescored = [
                    {**item, "score": dataclasses.replace(item["score"])}
                    for item in results
                ]
                rescored = blend_scores(rescored, adjusted_risk)
                scoring_note = (
                    f"weights adjusted: w_cvar={adjusted_risk.w_cvar:.2f} "
                    f"(+{w_cvar_boost:.2f}), w_opp={adjusted_risk.w_opportunity:.2f} "
                    f"(+{w_opp_boost:.2f})"
                )
            else:
                rescored = results
                scoring_note = "weights unchanged"

            # Step 2 — LLM picks winner from re-scored table.
            # Raw factors shown in $M for context; blendIdx is the unitless
            # normalized multi-criteria score actually used for ranking (lower=better).
            rows = ["idx | strategy | hedge% | mean$M | CVaR$M | opp$M | exec$M | blendIdx"]
            for idx, item in enumerate(rescored):
                p = item["params"]
                s = item["score"]
                rows.append(
                    f"{idx:3d} | {p.strategy_type.value:10s} | "
                    f"{_selected_hedge_fraction(p):6.0%} | "
                    f"{s.cost/1e6:7.2f} | {s.cvar/1e6:7.2f} | "
                    f"{s.opportunity_cost/1e6:7.2f} | {s.execution_risk/1e6:7.2f} | "
                    f"{s.blended:8.3f}"
                )

            prompt = (
                "You are a crude oil procurement risk advisor.\n\n"
                f"Market context: trend={market_ctx.get('price_trend','unknown')}, "
                f"vol_regime={market_ctx.get('vol_regime','unknown')}, "
                f"key_risks={market_ctx.get('key_risks', [])}.\n"
                f"Risk weight adjustment: {scoring_note}.\n\n"
                f"Candidates (re-scored for current regime):\n{chr(10).join(rows)}\n\n"
                f"Client base weights: CVaR={risk.w_cvar:.2f}, "
                f"max hedge={risk.max_hedge:.0%}\n\n"
                "The candidate at index 0 has the lowest blended score after regime adjustment.\n"
                "Select index 0 UNLESS there is a strong, specific regime reason to deviate "
                "(e.g. extreme volatility spike favours a trigger strategy already ranked close to top).\n"
                "If you deviate, explain exactly why in REASON.\n"
                "Respond exactly:\n"
                "REASON: <one sentence incorporating market context>\n"
                "INDEX: <integer>"
            )

            content = _llm_invoke(llm, prompt, max_tokens=256)
            idx_line = next((l for l in content.splitlines() if l.startswith("INDEX:")), None)
            reason_line = next((l for l in content.splitlines() if l.startswith("REASON:")), None)

            if idx_line:
                idx = max(0, min(int(idx_line.replace("INDEX:", "").strip()), len(rescored) - 1))
                best = rescored[idx]
                reason = reason_line.replace("REASON:", "").strip() if reason_line else ""
                msg = (
                    f"[node_arbitrate/LLM] idx={idx}, "
                    f"{_strategy_display(best['params'])}. "
                    f"{scoring_note}. {reason}"
                )

        except Exception as exc:
            logger.warning("node_arbitrate: LLM failed (%s); deterministic fallback.", exc)
            best = None

    if best is None:
        best = results[0]
        msg = (
            f"[node_arbitrate/det] {_strategy_display(best['params'])}, "
            f"blendIdx={best['score'].blended:.3f}."
        )

    best_params = best["params"]
    for record in records:
        if record.params == best_params:
            record.accepted = True
            record.note = record.note + " | Accepted as best candidate."
            break

    logger.info("%s", msg)
    return {
        "best": best,
        "records": records,
        "adjusted_risk": adjusted_risk,
        "trace_messages": state.get("trace_messages", []) + [msg],
    }


# ---------------------------------------------------------------------------
# Node: explain
# ---------------------------------------------------------------------------

def node_explain(state: AgentState) -> dict:
    best: dict = state["best"]
    forecast_obj = state["forecast_obj"]
    exposure: ExposureBook = state["exposure"]
    risk: RiskAppetite = state["risk"]
    forward_price = state["forward_price"]
    records: list[CandidateRecord] = state["records"]
    no_hedge_cost = state["no_hedge_cost"]
    market_ctx: dict = state.get("market_context", {})

    best_params = best["params"]
    best_cost = best["cost"]
    best_score = best["score"]

    policy = build_policy(params=best_params, forecast_obj=forecast_obj)

    savings_vs_no_hedge = no_hedge_cost.mean - best_cost.mean
    pct_savings = (
        savings_vs_no_hedge / no_hedge_cost.mean * 100.0
        if no_hedge_cost.mean != 0 else 0.0
    )

    llm = _get_llm()
    rationale = ""

    if llm:
        try:
            prompt = (
                "Write a 2-3 sentence recommendation for a crude oil procurement executive. "
                "Plain English, no markdown.\n\n"
                f"Market context: trend={market_ctx.get('price_trend','unknown')}, "
                f"vol_regime={market_ctx.get('vol_regime','unknown')}, "
                f"risks={market_ctx.get('key_risks',[])}.\n"
                f"Chosen strategy: {best_params.strategy_type.value}\n"
                f"Representative hedge: {_selected_hedge_fraction(best_params):.0%}\n"
                f"Expected cost: ${best_cost.mean/1e6:.2f}M\n"
                f"CVaR{int(risk.cvar_alpha*100)}: ${best_cost.cvar/1e6:.2f}M\n"
                f"Savings vs no hedge: ${savings_vs_no_hedge/1e6:.2f}M ({pct_savings:.1f}%)\n"
                f"Forward price: ${float(forward_price):.2f}/bbl\n"
                f"Schedule: {[round(float(f),2) for f in policy.hedge_fractions]}\n\n"
                "End with a clear action statement."
            )
            rationale = _llm_invoke(llm, prompt, max_tokens=256)
            msg = "[node_explain/LLM] LLM-written rationale."
        except Exception as exc:
            logger.warning("node_explain: LLM failed (%s); template fallback.", exc)
            rationale = ""

    if not rationale:
        rationale = (
            f"Recommend {_strategy_display(best_params)}. "
            f"Expected procurement cost ${best_cost.mean:,.0f}, "
            f"CVaR{int(risk.cvar_alpha*100)} ${best_cost.cvar:,.0f}. "
            f"Saves ${savings_vs_no_hedge:,.0f} ({pct_savings:.1f}%) vs no hedge."
        )
        msg = "[node_explain/det] Template rationale."

    paths = np.asarray(forecast_obj.paths, dtype=float)
    fan = _build_fan_data(paths)
    cost_histogram = _build_cost_histogram(
        strategy_costs=np.asarray(best_cost.costs, dtype=float),
        no_hedge_costs=np.asarray(no_hedge_cost.costs, dtype=float),
        cvar=best_cost.cvar,
    )

    recommendation = Recommendation(
        policy=policy,
        cost=best_cost,
        score=best_score,
        rationale=rationale,
        trace=records,
        assumptions={
            "forward_price": forward_price,
            "forecast_model": forecast_obj.model_name,
            "frequency": forecast_obj.frequency,
            "n_paths": forecast_obj.n_paths,
            "fan": fan,
            "cost_histogram": cost_histogram,
            "calibration_window": forecast_obj.calibration_window,
            "risk_weights": {
                "w_cost": risk.w_cost,
                "w_cvar": risk.w_cvar,
                "w_opportunity": risk.w_opportunity,
                "w_execution": risk.w_execution,
            },
            "market_context": market_ctx,
            "selected_strategy_type": best_params.strategy_type.value,
            "trace_messages": state.get("trace_messages", []),
            "forecast_obj": forecast_obj,
        },
    )

    logger.info("%s", msg)
    return {
        "recommendation": recommendation,
        "llm_rationale": rationale,
        "trace_messages": state.get("trace_messages", []) + [msg],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    """
    Compile LangGraph workflow.

    Parallel entry: forecast and market_intel run concurrently from START.
    Fan-out explore: each strategy family evaluated in its own node via Send API.
    """
    builder = StateGraph(AgentState)

    builder.add_node("forecast", node_forecast)
    builder.add_node("market_intel", node_market_intel)
    builder.add_node("assess", node_assess)
    builder.add_node("evaluate_family", node_evaluate_family)
    builder.add_node("explore_collect", node_explore_collect)
    builder.add_node("arbitrate", node_arbitrate)
    builder.add_node("explain", node_explain)

    # Parallel entry
    builder.add_edge(START, "forecast")
    builder.add_edge(START, "market_intel")

    # Both parallel nodes must complete before assess
    builder.add_edge("forecast", "assess")
    builder.add_edge("market_intel", "assess")

    # Fan-out: assess → evaluate_family via Send (node_explore_dispatch is the routing fn)
    builder.add_conditional_edges("assess", node_explore_dispatch, ["evaluate_family"])

    # Fan-in: each evaluate_family → collect
    builder.add_edge("evaluate_family", "explore_collect")

    builder.add_edge("explore_collect", "arbitrate")
    builder.add_edge("arbitrate", "explain")
    builder.add_edge("explain", END)

    return builder.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(
    history: PriceHistory,
    exposure: ExposureBook,
    risk: RiskAppetite,
    forward_price,
    frequency: str = "M",
    n_paths: int = 4096,
    seed: int = 42,
    calibration_window: int | None = 1000,
    distribution: str = "normal",
    use_regime: bool = False,
    model: str = "xgb-garch-t",
) -> Recommendation:
    initial = _initial_state(
        history=history,
        exposure=exposure,
        risk=risk,
        forward_price=forward_price,
        frequency=frequency,
        n_paths=n_paths,
        seed=seed,
        calibration_window=calibration_window,
        distribution=distribution,
        use_regime=use_regime,
        model=model,
    )
    final_state = _graph.invoke(initial)
    return final_state["recommendation"]
