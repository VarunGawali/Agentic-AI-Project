"""
Walk-forward backtester -- offline validation harness.

This is NOT part of live recommendation flow.

Purpose:
    Replay history with no look-ahead and test whether the selected hedge
    policy would have improved realized procurement cost versus baselines.

Phase 1:
    Uses current quant pipeline:
        train history -> forecast -> scorer picks best staggered hedge ratio
        -> realized cost evaluated on actual future prices

Baselines:
    - no hedge
    - naive 50% hedge
    - perfect foresight
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hedging_assistant.contracts import (
    PriceHistory,
    ExposureBook,
    RiskAppetite,
    BacktestResult,
)

from hedging_assistant.engines.forecaster import forecast, prepare_price_series
from hedging_assistant.engines.scorer import find_best_staggered_hedge
from hedging_assistant.engines.strategy_library import build_policy


def _realized_cost(
    hedge_fractions: np.ndarray,
    volumes: np.ndarray,
    actual_prices: np.ndarray,
    forward_price: float,
) -> float:
    """
    Realized cost of a hedge policy against actual future prices.
    """

    hedge_fractions = np.asarray(hedge_fractions, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    actual_prices = np.asarray(actual_prices, dtype=float)

    return float(
        np.sum(
            hedge_fractions * volumes * forward_price
            + (1.0 - hedge_fractions) * volumes * actual_prices
        )
    )


def _perfect_foresight_cost(
    volumes: np.ndarray,
    actual_prices: np.ndarray,
    forward_price: float,
) -> float:
    """
    Perfect foresight benchmark.

    Assumes if future spot is above forward, hedge 100%.
    If future spot is below forward, hedge 0%.

    This is not achievable in real life; it is only an upper benchmark.
    """

    volumes = np.asarray(volumes, dtype=float)
    actual_prices = np.asarray(actual_prices, dtype=float)

    best_prices = np.minimum(actual_prices, forward_price)

    return float(np.sum(volumes * best_prices))


def walk_forward_validate(
    history: PriceHistory,
    exposure: ExposureBook,
    risk: RiskAppetite,
    train_size: int = 60,
    step_size: int = 3,
    frequency: str = "M",
    n_paths: int = 10_000,
    seed: int | None = 42,
    calibration_window: int | None = 60,
    hedge_ratios: list[float] | None = None,
    forward_price_fn=None,
    label: str = "GBM baseline",
) -> BacktestResult:
    """
    Walk-forward validation.

    At each decision point:
        1. Use only past prices.
        2. Forecast future price paths.
        3. Pick best staggered hedge ratio using scorer.
        4. Compare selected policy against actual future prices.
        5. Store realized strategy/no-hedge/naive/perfect-foresight costs.

    Parameters:
        train_size:
            Number of historical observations in selected frequency.
            Example: 60 monthly observations.

        step_size:
            How far to move after each decision.
            Example: 3 means quarterly decision points on monthly data.

        frequency:
            D / W / M. Should match exposure period.
    """

    frequency = frequency.upper()

    if frequency not in {"D", "W", "M"}:
        raise ValueError("frequency must be one of: D, W, M")

    if train_size <= 1:
        raise ValueError("train_size must be greater than 1")

    if step_size <= 0:
        raise ValueError("step_size must be positive")

    horizon = exposure.horizon
    volumes = np.asarray(exposure.volumes, dtype=float)

    if horizon <= 0:
        raise ValueError("exposure horizon must be positive")

    # Default forward price = last known price at decision date
    if forward_price_fn is None:
        forward_price_fn = lambda train_prices: float(train_prices[-1])

    
    if calibration_window is not None and train_size <= calibration_window:
        raise ValueError(
            f"train_size must be greater than calibration_window. "
            f"Got train_size={train_size}, calibration_window={calibration_window}. "
            f"Reason: {calibration_window} returns require "
            f"{calibration_window + 1} price observations."
        )

    # Resample full history to backtest frequency
    full_series = prepare_price_series(history, frequency=frequency)

    all_dates = full_series.index.to_numpy()
    all_prices = full_series.to_numpy(dtype=float)

    n = len(all_prices)

    if n < train_size + horizon:
        raise ValueError(
            f"Not enough {frequency} observations for backtest. "
            f"Need at least train_size + horizon = {train_size + horizon}, got {n}."
        )

    decision_indices = list(range(train_size, n - horizon + 1, step_size))

    if not decision_indices:
        raise ValueError("No valid decision windows found.")

    strategy_costs = []
    perfect_foresight_costs = []
    no_hedge_costs = []
    naive_costs = []
    decision_dates = []

    selected_hedge_ratios = []

    for decision_idx in decision_indices:
        train_prices = all_prices[decision_idx - train_size: decision_idx]
        train_dates = all_dates[decision_idx - train_size: decision_idx]

        actual_prices = all_prices[decision_idx: decision_idx + horizon]

        train_history = PriceHistory(
            dates=train_dates,
            prices=train_prices,
            symbol=history.symbol,
        )

        forward_price = float(forward_price_fn(train_prices))

        # 1. Forecast using only past data
        forecast_obj = forecast(
            history=train_history,
            horizon=horizon,
            frequency=frequency,
            n_paths=n_paths,
            seed=seed,
            calibration_window=calibration_window,
            use_cache=False,
        )

        # 2. Pick best hedge ratio using current scorer
        best, _ = find_best_staggered_hedge(
            forecast_obj=forecast_obj,
            exposure=exposure,
            forward_price=forward_price,
            risk=risk,
            hedge_ratios=hedge_ratios,
        )

        selected_params = best["params"]
        selected_hedge_ratios.append(selected_params.base_fraction)

        policy = build_policy(
            params=selected_params,
            forecast_obj=forecast_obj,
        )

        hedge_fractions = policy.hedge_fractions

        # 3. Realized strategy cost
        strat_cost = _realized_cost(
            hedge_fractions=hedge_fractions,
            volumes=volumes,
            actual_prices=actual_prices,
            forward_price=forward_price,
        )

        # 4. No hedge baseline
        no_hedge_cost = float(np.sum(volumes * actual_prices))

        # 5. Naive 50% hedge baseline
        naive_fractions = np.full(horizon, 0.50, dtype=float)

        naive_cost = _realized_cost(
            hedge_fractions=naive_fractions,
            volumes=volumes,
            actual_prices=actual_prices,
            forward_price=forward_price,
        )

        # 6. Perfect foresight baseline
        pf_cost = _perfect_foresight_cost(
            volumes=volumes,
            actual_prices=actual_prices,
            forward_price=forward_price,
        )

        strategy_costs.append(strat_cost)
        no_hedge_costs.append(no_hedge_cost)
        naive_costs.append(naive_cost)
        perfect_foresight_costs.append(pf_cost)
        decision_dates.append(all_dates[decision_idx])

    strategy_costs = np.asarray(strategy_costs, dtype=float)
    no_hedge_costs = np.asarray(no_hedge_costs, dtype=float)
    naive_costs = np.asarray(naive_costs, dtype=float)
    perfect_foresight_costs = np.asarray(perfect_foresight_costs, dtype=float)
    decision_dates = np.asarray(decision_dates)

    mean_strategy = float(np.mean(strategy_costs))
    mean_no_hedge = float(np.mean(no_hedge_costs))
    mean_naive = float(np.mean(naive_costs))
    mean_pf = float(np.mean(perfect_foresight_costs))
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

    denominator = mean_no_hedge - mean_pf

    if abs(denominator) < 1e-9:
        captured_fraction = 0.0
    else:
        captured_fraction = float(
            np.clip(
                1.0 - (mean_strategy - mean_pf) / denominator,
                0.0,
                1.0,
            )
        )

    print(f"\n=== Walk-Forward Backtest: {label} ===")
    print(f"Frequency         : {frequency}")
    print(f"Windows           : {len(decision_indices)}")
    print(f"Train size        : {train_size}")
    print(f"Horizon           : {horizon}")
    print(f"Step size         : {step_size}")
    print(f"Mean strategy     : ${mean_strategy:,.2f}")
    print(f"Mean no-hedge     : ${mean_no_hedge:,.2f}")
    print(f"Mean naive 50%    : ${mean_naive:,.2f}")
    print(f"Mean perfect fs   : ${mean_pf:,.2f}")

    print(f"Savings vs no-hedge: ${savings_vs_no_hedge:,.2f}")
    print(f"Reduction vs no-hedge: {pct_savings_vs_no_hedge:.2f}%")

    print(f"Savings vs naive 50%: ${savings_vs_naive:,.2f}")
    print(f"Reduction vs naive 50%: {pct_savings_vs_naive:.2f}%")

    print(f"Captured fraction : {captured_fraction:.1%}")
    print(f"Avg selected hedge: {np.mean(selected_hedge_ratios):.0%}")


    return BacktestResult(
        dates=decision_dates,
        strategy_costs=strategy_costs,
        perfect_foresight_costs=perfect_foresight_costs,
        no_hedge_costs=no_hedge_costs,
        naive_costs=naive_costs,
        captured_fraction=captured_fraction,
        label=label,
    )