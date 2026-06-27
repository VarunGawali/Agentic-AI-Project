import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hedging_assistant.contracts import PriceHistory, ExposureBook, RiskAppetite
from hedging_assistant.backtest.validate import walk_forward_validate


def main():
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
        label="baseline_gbm_sobol_staggered_2factor",
    )

    strategy_costs = np.asarray(result.strategy_costs, dtype=float)
    no_hedge_costs = np.asarray(result.no_hedge_costs, dtype=float)
    naive_costs = np.asarray(result.naive_costs, dtype=float)
    perfect_foresight_costs = np.asarray(result.perfect_foresight_costs, dtype=float)

    mean_strategy = float(strategy_costs.mean())
    mean_no_hedge = float(no_hedge_costs.mean())
    mean_naive = float(naive_costs.mean())
    mean_pf = float(perfect_foresight_costs.mean())

    savings_vs_no_hedge = mean_no_hedge - mean_strategy
    savings_vs_naive = mean_naive - mean_strategy

    pct_savings_vs_no_hedge = (
        savings_vs_no_hedge / mean_no_hedge * 100
        if mean_no_hedge != 0
        else 0.0
    )

    pct_savings_vs_naive = (
        savings_vs_naive / mean_naive * 100
        if mean_naive != 0
        else 0.0
    )

    benchmark = {
        "benchmark_name": "baseline_gbm_sobol_staggered_2factor",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data": {
            "symbol": "WTI",
            "source": "local_csv_or_blob_snapshot",
            "first_date": str(df["date"].min().date()),
            "last_date": str(df["date"].max().date()),
            "rows": int(len(df)),
        },
        "forecast_config": {
            "model": "GBM-Sobol-Antithetic",
            "frequency": "M",
            "horizon": int(exposure.horizon),
            "n_paths": 8192,
            "seed": 42,
            "calibration_window": 60,
        },
        "strategy_config": {
            "strategy_universe": ["STAGGERED"],
            "hedge_ratios": [0.0, 0.25, 0.50, 0.75, 1.0],
        },
        "risk_config": {
            "w_cost": risk.w_cost,
            "w_cvar": risk.w_cvar,
            "w_opportunity": risk.w_opportunity,
            "w_execution": risk.w_execution,
            "cvar_alpha": risk.cvar_alpha,
            "max_hedge": risk.max_hedge,
        },
        "backtest_config": {
            "train_size": 72,
            "step_size": 3,
            "frequency": "M",
            "windows": int(len(result.dates)),
        },
        "metrics": {
            "mean_strategy_cost": mean_strategy,
            "mean_no_hedge_cost": mean_no_hedge,
            "mean_naive_50_cost": mean_naive,
            "mean_perfect_foresight_cost": mean_pf,
            "savings_vs_no_hedge": savings_vs_no_hedge,
            "savings_vs_naive_50": savings_vs_naive,
            "pct_savings_vs_no_hedge": pct_savings_vs_no_hedge,
            "pct_savings_vs_naive_50": pct_savings_vs_naive,
            "captured_fraction": float(result.captured_fraction),
        },
        "arrays": {
            "dates": [str(d) for d in result.dates],
            "strategy_costs": strategy_costs.tolist(),
            "no_hedge_costs": no_hedge_costs.tolist(),
            "naive_costs": naive_costs.tolist(),
            "perfect_foresight_costs": perfect_foresight_costs.tolist(),
        },
        "notes": [
            "This is the pre-Phase-3 benchmark.",
            "Use this to compare GARCH-t, trigger strategies, volatility strategies, four-factor scoring, and Optuna search.",
        ],
    }

    output_dir = Path("benchmarks")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "baseline_gbm_sobol_staggered_2factor.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)

    print("\nSaved baseline benchmark")
    print("------------------------")
    print(f"Path                  : {output_path}")
    print(f"Windows               : {len(result.dates)}")
    print(f"Mean strategy cost    : ${mean_strategy:,.2f}")
    print(f"Mean no-hedge cost    : ${mean_no_hedge:,.2f}")
    print(f"Mean naive 50% cost   : ${mean_naive:,.2f}")
    print(f"Mean perfect foresight: ${mean_pf:,.2f}")
    print(f"Savings vs no hedge   : ${savings_vs_no_hedge:,.2f}")
    print(f"Reduction vs no hedge : {pct_savings_vs_no_hedge:.2f}%")
    print(f"Captured fraction     : {result.captured_fraction:.4f}")


if __name__ == "__main__":
    main()