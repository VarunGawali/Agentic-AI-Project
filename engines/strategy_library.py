"""
Strategy library: generate HedgingPolicy hedge fractions from StrategyParams.

BASELINE (Phase 1): staggered strategy with cap constraint + candidate generator.
UPGRADE  (Phase 3): trigger-based and volatility-based (path-dependent).
"""

from __future__ import annotations
import numpy as np

from hedging_assistant.contracts import (
    StrategyType, StrategyParams, HedgingPolicy, PriceForecast,
)


def apply_strategy(
    params: StrategyParams,
    price_path: np.ndarray,
    history_vol: float | None = None,
) -> np.ndarray:
    """
    Apply a hedging strategy to a single price path.

    Args:
        params: strategy parameters
        price_path: shape (horizon,) — one simulated price path
        history_vol: recent historical volatility (used by VOLATILITY strategy)

    Returns:
        hedge fractions shape (horizon,), each in [0, cap]
    """
    horizon = len(price_path)
    if horizon == 0:
        raise ValueError("price_path is empty.")

    if params.strategy_type == StrategyType.STAGGERED:
        raw = np.full(horizon, params.base_fraction)
        # apply cap constraint — hedge fraction never exceeds params.cap
        fractions = np.clip(raw, 0.0, params.cap)
        return fractions

    if params.strategy_type == StrategyType.TRIGGER:
        raise NotImplementedError("Phase 3: hedge more when price crosses trigger_price")

    if params.strategy_type == StrategyType.VOLATILITY:
        raise NotImplementedError("Phase 3: hedge more when recent vol exceeds vol_threshold")

    raise ValueError(f"Unknown strategy type: {params.strategy_type}")


def build_policy(params: StrategyParams, forecast_obj: PriceForecast) -> HedgingPolicy:
    """Build a representative HedgingPolicy using the median forecast path."""
    median_path = np.percentile(forecast_obj.paths, 50, axis=0)
    fractions = apply_strategy(params, median_path)
    desc = (
        f"{params.base_fraction:.0%} {params.strategy_type.value} "
        f"(cap={params.cap:.0%})"
    )
    return HedgingPolicy(params=params, hedge_fractions=fractions, description=desc)


def generate_staggered_candidates(
    max_hedge: float = 1.0,
    n_steps: int = 11,
    cap: float = 1.0,
) -> list[StrategyParams]:
    """
    Generate a grid of staggered strategy candidates.

    Args:
        max_hedge: maximum allowed hedge fraction
        n_steps: number of evenly-spaced fractions from 0 to max_hedge
        cap: cap constraint applied inside apply_strategy

    Returns:
        list of StrategyParams, one per candidate
    """
    if max_hedge <= 0 or max_hedge > 1.0:
        raise ValueError(f"max_hedge must be in (0, 1]; got {max_hedge}")
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2; got {n_steps}")

    fractions = np.linspace(0.0, max_hedge, n_steps)
    candidates = []
    for frac in fractions:
        candidates.append(
            StrategyParams(
                strategy_type=StrategyType.STAGGERED,
                base_fraction=float(frac),
                cap=float(cap),
            )
        )
    return candidates


def generate_batch_candidates(
    strategy_types: list[StrategyType] | None = None,
    max_hedge: float = 1.0,
    n_steps: int = 11,
    cap: float = 1.0,
) -> list[StrategyParams]:
    """
    Generate candidates across multiple strategy types (batch creation).
    Phase 3 trigger/volatility types raise NotImplementedError until implemented.

    Args:
        strategy_types: list of StrategyType to include; defaults to [STAGGERED]
        max_hedge: maximum hedge fraction
        n_steps: grid resolution for staggered
        cap: cap constraint

    Returns:
        flat list of all StrategyParams candidates
    """
    if strategy_types is None:
        strategy_types = [StrategyType.STAGGERED]

    all_candidates: list[StrategyParams] = []
    for stype in strategy_types:
        if stype == StrategyType.STAGGERED:
            all_candidates.extend(
                generate_staggered_candidates(max_hedge, n_steps, cap)
            )
        else:
            raise NotImplementedError(f"Candidate generation for {stype} is Phase 3.")
    return all_candidates
