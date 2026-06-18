"""
Test script: verifies the LangGraph plan-and-execute workflow runs end-to-end.
"""
import sys, os
# The hedging_assistant namespace package lives in the shared checkout root
_shared = "/home/user/Agentic-AI-Project"
if _shared not in sys.path:
    sys.path.insert(0, _shared)

from hedging_assistant.data.loader import load_price_history, make_exposure_book, default_risk_appetite
from hedging_assistant.agent.langgraph_agent import run_agent
from hedging_assistant.agent.orchestrator import HedgingAgent

history  = load_price_history(symbol="WTI")
exposure = make_exposure_book(barrels_per_period=100_000, horizon=6)
risk     = default_risk_appetite()
fwd      = float(history.prices[-1])

print("=" * 60)
print(" TEST 1: run_agent() direct LangGraph call")
print("=" * 60)
rec = run_agent(history, exposure, risk, fwd)
print("\nRationale:\n" + rec.rationale)

print("\n" + "=" * 60)
print(" TEST 2: HedgingAgent(use_langgraph=True).recommend()")
print("=" * 60)
rec2 = HedgingAgent(risk, use_langgraph=True).recommend(history, exposure, fwd)
print(f"\nPolicy: hedge {rec2.policy.params.base_fraction:.0%}")
print(f"Model : {rec2.assumptions['model']}")
print("\nAll tests passed.")
