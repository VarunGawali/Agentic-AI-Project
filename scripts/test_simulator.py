import numpy as np
import pandas as pd

from hedging_assistant.contracts import (
    PriceHistory,
    ExposureBook,
    StrategyParams,
    StrategyType,
    RiskAppetite,
)

from hedging_assistant.engines.forecaster import forecast
from hedging_assistant.engines.cost_simulator import simulate_cost
from hedging_assistant.engines.scorer import find_best_staggered_hedge


# ---------------------------------------------------------
# 1. Load historical WTI price data
# ---------------------------------------------------------

df = pd.read_csv("data/raw/wti_price_history.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

history = PriceHistory(
    dates=df["date"].to_numpy(),
    prices=df["price"].to_numpy(dtype=float),
    symbol="WTI",
)


# ---------------------------------------------------------
# 2. Generate forecast_obj
# ---------------------------------------------------------

forecast_obj = forecast(
    history=history,
    horizon=6,
    frequency="M",
    n_paths=10_000,
    seed=42,
    calibration_window=60,
    use_cache=True,
)


# ---------------------------------------------------------
# 3. User-provided exposure book
# ---------------------------------------------------------

exposure = ExposureBook(
    volumes=np.array(
        [100_000, 100_000, 100_000, 100_000, 100_000, 100_000],
        dtype=float,
    ),
    period_label="month",
)


# ---------------------------------------------------------
# 4. Single strategy test: 50% staggered hedge
# ---------------------------------------------------------

params = StrategyParams(
    strategy_type=StrategyType.STAGGERED,
    base_fraction=0.50,
    cap=1.0,
)

cost_dist = simulate_cost(
    forecast_obj=forecast_obj,
    exposure=exposure,
    params=params,
    forward_price=95.0,
    cvar_alpha=0.95,
    mode="optimized",
    compute_ci=False,
)


print("\nSingle Strategy Cost Simulation")
print("-------------------------------")
print("Strategy    : 50% staggered hedge")
print(f"Mean cost   : ${cost_dist.mean:,.2f}")
print(f"P10 cost    : ${cost_dist.p10:,.2f}")
print(f"P50 cost    : ${cost_dist.p50:,.2f}")
print(f"P90 cost    : ${cost_dist.p90:,.2f}")
print(f"CVaR95 cost : ${cost_dist.cvar:,.2f}")


# ---------------------------------------------------------
# 5. Hedge-ratio optimization
# ---------------------------------------------------------

risk = RiskAppetite(
    w_cost=1.0,
    w_cvar=0.25,
    cvar_alpha=0.95,
    max_hedge=1.0,
)

best, results = find_best_staggered_hedge(
    forecast_obj=forecast_obj,
    exposure=exposure,
    forward_price=95.0,
    risk=risk,
    hedge_ratios=[0.0, 0.25, 0.50, 0.75, 1.0],
)


# ---------------------------------------------------------
# 6. Print ranked results
# ---------------------------------------------------------

print("\nHedge Ratio Comparison")
print("----------------------")

# Print in hedge-ratio order for readability
for r in sorted(results, key=lambda x: x["hedge_ratio"]):
    ratio = r["hedge_ratio"]
    cost = r["cost"]
    score = r["score"]

    print(f"\nHedge ratio : {ratio:.0%}")
    print(f"Mean cost   : ${cost.mean:,.2f}")
    print(f"P90 cost    : ${cost.p90:,.2f}")
    print(f"CVaR95 cost : ${cost.cvar:,.2f}")
    print(f"Score       : {score.blended:,.2f}")


print("\nRecommended Hedge Policy")
print("------------------------")
print(f"Best hedge ratio : {best['hedge_ratio']:.0%}")
print(f"Expected cost    : ${best['cost'].mean:,.2f}")
print(f"CVaR95 cost      : ${best['cost'].cvar:,.2f}")
print(f"Blended score    : {best['score'].blended:,.2f}")