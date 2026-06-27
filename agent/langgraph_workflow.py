"""
LangGraph plan-and-execute workflow for the hedging assistant.

Workflow:
    assess -> forecast -> explore -> arbitrate -> explain

Phase 3:
    - Default forecaster: XGB-GARCH-t
    - Candidate universe:
        STAGGERED
        TRIGGER
        VOLATILITY
        HYBRID
        CVAR_LP optimizer-generated schedule
    - Scoring:
        Expected Cost + CVaR + Opportunity Cost + Execution Risk
    - Optional LLM nodes:
        assess / arbitrate / explain use Azure OpenAI if env vars are set.
        Otherwise deterministic Python fallback is used.

Optional environment variables:
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_DEPLOYMENT
    AZURE_OPENAI_API_VERSION
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import AzureOpenAI

import numpy as np
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from contracts import (
    PriceHistory,
    ExposureBook,
    RiskAppetite,
    StrategyType,
    StrategyParams,
    CandidateRecord,
    Recommendation,
)

from engines.forecaster import forecast
from engines.strategy_library import (
    build_policy,
    generate_batch_candidates,
)
from engines.cost_simulator import simulate_cost
from engines.scorer import evaluate_candidates


import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")



# ---------------------------------------------------------------------------
# Optional LLM — lazy init, graceful fallback
# ---------------------------------------------------------------------------

LLM_SYSTEM_INSTRUCTIONS = """
You are a financial decision-support assistant for a crude oil procurement and hedging dashboard.

You do not perform numerical forecasting, optimization, Monte Carlo simulation, scoring, or CVaR calculation yourself. Those are handled by deterministic Python engines.

Your role is limited to:
1. Assessing market/risk context and suggesting which strategy families should be evaluated.
2. Reasoning over already-computed candidate strategy scores.
3. Writing concise, business-friendly explanations for procurement executives.

Important rules:
- Never invent prices, costs, hedge ratios, CVaR values, or candidate results.
- Use only the numbers and candidate table provided in the prompt.
- If asked to choose a strategy, choose only from the provided candidate indices.
- Prefer the lowest blended score unless there is a clear risk-weight tradeoff explained in the prompt.
- Keep responses short, structured, and deterministic.
- For JSON requests, respond with valid JSON only.
- For arbitration requests, respond exactly in the requested format.
- For executive explanations, use plain English, no markdown, no bullets, and include the chosen strategy, expected cost, CVaR, and action recommendation.
"""


def _get_llm():
    """
    Initialize Azure OpenAI client only if env vars are configured.

    If anything fails, return None and use deterministic fallback.
    """

    key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not key or not endpoint:
        return None

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )

        return client

    except Exception as exc:
        print(f"[langgraph_agent] LLM init failed ({exc}); deterministic fallback.")
        return None
    
def _llm_invoke(
    client,
    prompt: str,
    max_tokens: int = 512,
) -> str:
    """
    Invoke Azure OpenAI chat completion using official OpenAI SDK.

    Uses the deployment name from AZURE_OPENAI_DEPLOYMENT.
    """

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

    response = client.chat.completions.create(
        model=deployment,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": LLM_SYSTEM_INSTRUCTIONS,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """
    State carried between LangGraph nodes.
    """

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

    candidates: list[StrategyParams]
    forecast_obj: Any
    results: list[dict]
    records: list[CandidateRecord]
    no_hedge_cost: Any
    best: dict
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
    """
    Build initial graph state.
    """

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
        "candidates": [],
        "forecast_obj": None,
        "results": [],
        "records": [],
        "no_hedge_cost": None,
        "best": {},
        "recommendation": None,
        "trace_messages": [],
        "llm_rationale": "",
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _selected_hedge_fraction(params: StrategyParams) -> float:
    """
    Representative hedge fraction for logs/rationale.
    """

    if params.strategy_type == StrategyType.CVAR_LP and params.fixed_fractions is not None:
        return float(np.mean(params.fixed_fractions))

    return float(params.base_fraction)


def _strategy_display(params: StrategyParams) -> str:
    """
    Human-readable strategy label.
    """

    hedge_pct = _selected_hedge_fraction(params) * 100.0

    if params.strategy_type == StrategyType.CVAR_LP:
        return f"CVaR-LP optimized schedule, avg hedge {hedge_pct:.0f}%"

    return f"{hedge_pct:.0f}% {params.strategy_type.value}"


def _build_fan_data(paths: np.ndarray) -> dict:
    """
    Build P10/P25/P50/P75/P90 fan chart data.
    """

    return {
        "p10": np.percentile(paths, 10, axis=0).tolist(),
        "p25": np.percentile(paths, 25, axis=0).tolist(),
        "p50": np.percentile(paths, 50, axis=0).tolist(),
        "p75": np.percentile(paths, 75, axis=0).tolist(),
        "p90": np.percentile(paths, 90, axis=0).tolist(),
    }


def _build_cost_histogram(
    strategy_costs: np.ndarray,
    no_hedge_costs: np.ndarray,
    cvar: float,
    bins: int = 40,
) -> dict:
    """
    Build dashboard cost histogram in millions.
    """

    strategy_counts, strategy_edges = np.histogram(strategy_costs / 1e6, bins=bins)
    no_hedge_counts, no_hedge_edges = np.histogram(no_hedge_costs / 1e6, bins=bins)

    return {
        "strategy": {
            "counts": strategy_counts.tolist(),
            "edges": strategy_edges.tolist(),
        },
        "no_hedge": {
            "counts": no_hedge_counts.tolist(),
            "edges": no_hedge_edges.tolist(),
        },
        "cvar_line": round(cvar / 1e6, 2),
    }


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_assess(state: AgentState) -> dict:
    """
    Assess inputs and generate candidate strategies.

    Deterministic fallback:
        Generate STAGGERED, TRIGGER, VOLATILITY, HYBRID candidates.

    Optional LLM:
        Can suggest whether to include aggressive/conservative candidate mix.
        We still produce StrategyParams deterministically.
    """

    risk: RiskAppetite = state["risk"]
    history: PriceHistory = state["history"]

    strategy_types = [
        StrategyType.STAGGERED,
        StrategyType.TRIGGER,
        StrategyType.VOLATILITY,
        StrategyType.HYBRID,
    ]

    llm = _get_llm()

    if llm:
        try:
            prices = np.asarray(history.prices, dtype=float)
            recent_prices = prices[-30:]

            recent_vol = float(
                np.std(np.diff(np.log(recent_prices)), ddof=1) * np.sqrt(252)
            )

            recent_trend = float((prices[-1] / prices[-20] - 1.0) * 100.0)

            prompt = (
                "You are a crude oil procurement risk advisor.\n"
                "Given the market context, choose the strategy families to evaluate.\n\n"
                f"Annualized recent volatility: {recent_vol:.2%}\n"
                f"20-day price trend: {recent_trend:+.2f}%\n"
                f"Client max hedge: {risk.max_hedge:.0%}\n"
                f"CVaR weight: {risk.w_cvar:.2f}\n"
                f"Opportunity weight: {risk.w_opportunity:.2f}\n"
                f"Execution weight: {risk.w_execution:.2f}\n\n"
                "Available strategy families: staggered, trigger, volatility, hybrid.\n"
                "Return ONLY a valid JSON array of strategy family strings."
            )

            raw = _llm_invoke(llm, prompt, max_tokens=256)

            if raw.startswith("```"):
                raw = raw.split("```")[1].replace("json", "").strip()

            selected = json.loads(raw)

            mapping = {
                "staggered": StrategyType.STAGGERED,
                "trigger": StrategyType.TRIGGER,
                "volatility": StrategyType.VOLATILITY,
                "hybrid": StrategyType.HYBRID,
            }

            parsed = [
                mapping[item.lower()]
                for item in selected
                if isinstance(item, str) and item.lower() in mapping
            ]

            if parsed:
                strategy_types = list(dict.fromkeys(parsed))

            msg_prefix = "[node_assess/LLM]"

        except Exception as exc:
            print(f"[node_assess] LLM failed ({exc}); deterministic fallback.")
            msg_prefix = "[node_assess/det]"
    else:
        msg_prefix = "[node_assess/det]"

    candidates = generate_batch_candidates(
        strategy_types=strategy_types,
        max_hedge=risk.max_hedge,
        n_steps=5,
        cap=risk.max_hedge,
    )

    msg = (
        f"{msg_prefix} Generated {len(candidates)} candidates from "
        f"{[strategy.value for strategy in strategy_types]}."
    )

    print(msg)

    return {
        "candidates": candidates,
        "trace_messages": state.get("trace_messages", []) + [msg],
    }


def node_forecast(state: AgentState) -> dict:
    """
    Generate forecast paths using configured model.

    Default model:
        xgb-garch-t
    """

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
        f"[node_forecast] Produced {forecast_obj.n_paths} paths x "
        f"{forecast_obj.horizon} steps using {forecast_obj.model_name}."
    )

    print(msg)

    return {
        "forecast_obj": forecast_obj,
        "trace_messages": state.get("trace_messages", []) + [msg],
    }


def node_explore(state: AgentState) -> dict:
    """
    Evaluate candidate hedge strategies.

    Adds CVaR-LP optimized schedule as an extra candidate if cvxpy is available.
    """

    forecast_obj = state["forecast_obj"]
    exposure: ExposureBook = state["exposure"]
    risk: RiskAppetite = state["risk"]
    forward_price = state["forward_price"]

    candidates = list(state.get("candidates", []))

    # ---------------------------------------------------------
    # Add CVaR-LP optimizer-generated candidate
    # ---------------------------------------------------------

    try:
        from engines.optimizer import optimize_cvar_lp_params

        cvar_lp_params = optimize_cvar_lp_params(
            forecast_obj=forecast_obj,
            exposure=exposure,
            forward_price=forward_price,
            cvar_alpha=risk.cvar_alpha,
            cost_weight=risk.w_cost,
            cvar_weight=risk.w_cvar,
            max_hedge=risk.max_hedge,
        )

        candidates.append(cvar_lp_params)

        lp_msg = "[node_explore] Added CVaR-LP optimized candidate."

    except Exception as exc:
        lp_msg = f"[node_explore] CVaR-LP skipped: {exc}"

    print(lp_msg)

    # ---------------------------------------------------------
    # No-hedge baseline
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Evaluate all candidates
    # ---------------------------------------------------------

    results = evaluate_candidates(
        forecast_obj=forecast_obj,
        exposure=exposure,
        risk=risk,
        forward_price=forward_price,
        candidates=candidates,
        mode="accurate",
        compute_ci=False,
    )

    records = []

    for item in results:
        params = item["params"]
        cost = item["cost"]
        score = item["score"]

        note = (
            f"Strategy={params.strategy_type.value}; "
            f"mean=${cost.mean:,.0f}; "
            f"CVaR=${cost.cvar:,.0f}; "
            f"opp=${score.opportunity_cost:,.0f}; "
            f"exec=${score.execution_risk:,.0f}; "
            f"blended=${score.blended:,.0f}"
        )

        records.append(
            CandidateRecord(
                params=params,
                score=score,
                accepted=False,
                note=note,
            )
        )

    msg = (
        f"[node_explore] Evaluated {len(results)} candidates. "
        f"No-hedge mean cost=${no_hedge_cost.mean:,.0f}."
    )

    print(msg)

    return {
        "candidates": candidates,
        "results": results,
        "records": records,
        "no_hedge_cost": no_hedge_cost,
        "trace_messages": state.get("trace_messages", []) + [lp_msg, msg],
    }


def node_arbitrate(state: AgentState) -> dict:
    """
    Select best candidate.

    LLM can reason over tradeoffs, but deterministic fallback chooses min score.
    """

    results: list[dict] = state["results"]
    records: list[CandidateRecord] = state["records"]
    risk: RiskAppetite = state["risk"]

    if not results:
        raise ValueError("No candidate results available for arbitration.")

    llm = _get_llm()
    best = None

    if llm:
        try:
            rows = [
                "idx | strategy | hedge% | mean$M | CVaR$M | opp$M | exec$M | blended$M"
            ]

            for idx, item in enumerate(results):
                params = item["params"]
                score = item["score"]

                rows.append(
                    f"{idx:3d} | "
                    f"{params.strategy_type.value:10s} | "
                    f"{_selected_hedge_fraction(params):6.0%} | "
                    f"{score.cost / 1e6:7.2f} | "
                    f"{score.cvar / 1e6:7.2f} | "
                    f"{score.opportunity_cost / 1e6:7.2f} | "
                    f"{score.execution_risk / 1e6:7.2f} | "
                    f"{score.blended / 1e6:7.2f}"
                )

            prompt = (
                "You are a crude oil procurement risk advisor.\n\n"
                "Choose the best strategy index. Lower blended score is usually best, "
                "but explain any tradeoff if selecting a different one.\n\n"
                f"Client weights: CVaR={risk.w_cvar:.2f}, "
                f"opportunity={risk.w_opportunity:.2f}, "
                f"execution={risk.w_execution:.2f}, "
                f"max hedge={risk.max_hedge:.0%}\n\n"
                f"{chr(10).join(rows)}\n\n"
                "Respond exactly:\n"
                "REASON: <one sentence>\n"
                "INDEX: <integer>"
            )

            content = _llm_invoke(llm, prompt, max_tokens=256)

            idx_line = next(
                (line for line in content.splitlines() if line.startswith("INDEX:")),
                None,
            )

            reason_line = next(
                (line for line in content.splitlines() if line.startswith("REASON:")),
                None,
            )

            if idx_line:
                idx = int(idx_line.replace("INDEX:", "").strip())
                idx = max(0, min(idx, len(results) - 1))
                best = results[idx]

                reason = (
                    reason_line.replace("REASON:", "").strip()
                    if reason_line
                    else ""
                )

                msg = (
                    f"[node_arbitrate/LLM] Selected idx={idx}, "
                    f"{_strategy_display(best['params'])}. {reason}"
                )

        except Exception as exc:
            print(f"[node_arbitrate] LLM failed ({exc}); deterministic fallback.")
            best = None

    if best is None:
        best = results[0]
        msg = (
            f"[node_arbitrate/det] Selected {_strategy_display(best['params'])}, "
            f"score={best['score'].blended:,.0f}."
        )

    best_params = best["params"]

    for record in records:
        if record.params == best_params:
            record.accepted = True
            record.note = record.note + " | Accepted as best candidate."
            break

    print(msg)

    return {
        "best": best,
        "records": records,
        "trace_messages": state.get("trace_messages", []) + [msg],
    }


def node_explain(state: AgentState) -> dict:
    """
    Build final Recommendation object.
    """

    best: dict = state["best"]
    forecast_obj = state["forecast_obj"]
    exposure: ExposureBook = state["exposure"]
    risk: RiskAppetite = state["risk"]
    forward_price = state["forward_price"]
    records: list[CandidateRecord] = state["records"]
    no_hedge_cost = state["no_hedge_cost"]

    best_params = best["params"]
    best_cost = best["cost"]
    best_score = best["score"]

    policy = build_policy(
        params=best_params,
        forecast_obj=forecast_obj,
    )

    savings_vs_no_hedge = no_hedge_cost.mean - best_cost.mean

    pct_savings_vs_no_hedge = (
        savings_vs_no_hedge / no_hedge_cost.mean * 100.0
        if no_hedge_cost.mean != 0
        else 0.0
    )

    llm = _get_llm()
    rationale = ""

    if llm:
        try:
            prompt = (
                "Write a 2-3 sentence recommendation for a crude oil procurement "
                "executive. Use plain English, no markdown.\n\n"
                f"Chosen strategy: {best_params.strategy_type.value}\n"
                f"Representative hedge: {_selected_hedge_fraction(best_params):.0%}\n"
                f"Expected cost: ${best_cost.mean / 1e6:.2f}M\n"
                f"CVaR{int(risk.cvar_alpha * 100)}: ${best_cost.cvar / 1e6:.2f}M\n"
                f"Opportunity cost: ${best_score.opportunity_cost / 1e6:.2f}M\n"
                f"Execution risk: ${best_score.execution_risk / 1e6:.2f}M\n"
                f"Forward price: ${float(forward_price):.2f}/bbl\n"
                f"Schedule: {[round(float(f), 2) for f in policy.hedge_fractions]}\n"
                f"Savings vs no hedge: ${savings_vs_no_hedge / 1e6:.2f}M\n\n"
                "End with a clear action statement."
            )

            rationale = _llm_invoke(llm, prompt, max_tokens=256)
            msg = "[node_explain/LLM] LLM-written rationale."

        except Exception as exc:
            print(f"[node_explain] LLM failed ({exc}); deterministic fallback.")
            rationale = ""

    if not rationale:
        rationale = (
            f"Recommend {_strategy_display(best_params)}. "
            f"The expected procurement cost is ${best_cost.mean:,.0f}, "
            f"with CVaR{int(risk.cvar_alpha * 100)} tail cost of "
            f"${best_cost.cvar:,.0f}. "
            f"Compared with no hedge, this changes expected cost by "
            f"${savings_vs_no_hedge:,.0f} "
            f"({pct_savings_vs_no_hedge:.2f}%). "
            f"The final blended score is {best_score.blended:,.0f}, "
            f"including expected cost, CVaR, opportunity cost, and execution risk."
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
            "selected_strategy_type": best_params.strategy_type.value,
            "trace_messages": state.get("trace_messages", []),
            # Useful for API fallback; api._json_safe removes this.
            "forecast_obj": forecast_obj,
        },
    )

    print(msg)

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
    """

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
    """
    Run full LangGraph workflow and return Recommendation.
    """

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