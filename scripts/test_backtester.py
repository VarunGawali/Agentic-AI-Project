import numpy as np
import pandas as pd

from hedging_assistant.contracts import PriceHistory, ExposureBook, RiskAppetite
from hedging_assistant.backtest.validate import walk_forward_validate


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
    cvar_alpha=0.95,
    max_hedge=1.0,
)

result = walk_forward_validate(
    history=history,
    exposure=exposure,
    risk=risk,
    train_size=72,
    step_size=3,
    frequency="M",
    n_paths=8192,
    seed=42,
    calibration_window=60,
    hedge_ratios=[0.0, 0.25, 0.50, 0.75, 1.0],
    label="GBM-Sobol baseline",
)

print("\nBacktest arrays")
print("---------------")
print("Dates:", result.dates[:5])
print("Strategy costs:", np.round(result.strategy_costs[:5], 2))
print("No hedge costs:", np.round(result.no_hedge_costs[:5], 2))
print("Naive costs:", np.round(result.naive_costs[:5], 2))
print("Perfect foresight costs:", np.round(result.perfect_foresight_costs[:5], 2))
print("Captured fraction:", round(result.captured_fraction, 4))