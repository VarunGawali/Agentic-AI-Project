import numpy as np
import pandas as pd

from contracts import PriceHistory, ExposureBook, RiskAppetite
from agent.langgraph_workflow import run_agent


df = pd.read_csv("data/raw/wti_price_history.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

history = PriceHistory(
    dates=df["date"].to_numpy(),
    prices=df["price"].to_numpy(dtype=float),
    symbol="WTI",
)

exposure = ExposureBook(
    volumes=np.array(
        [100_000, 100_000, 100_000, 100_000, 100_000, 100_000],
        dtype=float,
    ),
    period_label="month",
)

risk = RiskAppetite(
    w_cost=1.0,
    w_cvar=0.25,
    w_opportunity=0.0,
    w_execution=0.0,
    cvar_alpha=0.95,
    max_hedge=1.0,
)

recommendation = run_agent(
    history=history,
    exposure=exposure,
    risk=risk,
    forward_price=95.0,
    frequency="M",
    n_paths=8192,
    seed=42,
    calibration_window=60,
    distribution="normal",
    use_regime=False,
)

print("\nFinal Recommendation")
print("--------------------")
print(recommendation.rationale)

print("\nPolicy")
print("------")
print(recommendation.policy.description)
print(recommendation.policy.hedge_fractions)

print("\nCost")
print("----")
print(f"Mean cost  : ${recommendation.cost.mean:,.2f}")
print(f"P90 cost   : ${recommendation.cost.p90:,.2f}")
print(f"CVaR cost  : ${recommendation.cost.cvar:,.2f}")

print("\nTrace")
print("-----")
for item in recommendation.trace:
    print(
        f"{item.params.base_fraction:.0%} hedge | "
        f"score={item.score.blended:,.2f} | "
        f"accepted={item.accepted} | "
        f"{item.note}"
    )