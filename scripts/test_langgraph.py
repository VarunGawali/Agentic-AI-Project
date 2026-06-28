"""
Test script: verifies the LangGraph agentic workflow runs end-to-end.
"""
import sys, os

_shared = "/home/user/Agentic-AI-Project"
if _shared not in sys.path:
    sys.path.insert(0, _shared)

from hedging_assistant.data.loader import load_price_history
from hedging_assistant.contracts import ExposureBook, RiskAppetite
from hedging_assistant.agent.langgraph_workflow import run_agent

history = load_price_history(symbol="WTI")

exposure = ExposureBook(
    volumes=[100_000] * 6,
    horizon=6,
)

risk = RiskAppetite(
    max_hedge=0.80,
    cvar_alpha=0.95,
    w_cost=0.5,
    w_cvar=0.3,
    w_opportunity=0.1,
    w_execution=0.1,
)

fwd = float(history.prices[-1])

print("=" * 60)
print(" TEST: run_agent() — agentic LangGraph workflow")
print("=" * 60)

rec = run_agent(history, exposure, risk, fwd)

print(f"\nStrategy : {rec.policy.params.strategy_type.value}")
print(f"Hedge    : {rec.policy.params.base_fraction:.0%}")
print(f"Expected cost: ${rec.cost.mean:,.0f}")
print(f"CVaR ({risk.cvar_alpha:.0%}): ${rec.cost.cvar:,.0f}")
print(f"\nRationale:\n{rec.rationale}")
print("\nAll tests passed.")
