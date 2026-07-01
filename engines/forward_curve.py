"""
Forward-curve construction for the hedging optimizer.

Why this matters
----------------
With `forward = spot` and a driftless (no-directional-skill) forecast,
E[spot_t] == F_t, so the expected value of hedging is exactly zero and no
strategy can beat no-hedge on expected cost. The *forward curve* is what
creates a real, observable basis:

    expected saving from hedging period t  =  volume_t * (E[spot_t] - F_t)

    F_t < E[spot_t]  (backwardation) -> hedging saves on average
    F_t > E[spot_t]  (contango)      -> hedging costs on average

The basis is quoted by the market today (it is data, not a forecast), so this
is an honest source of expected savings — unlike predicting price direction.

Sources
-------
1. Parametric (scenario / fallback):  F_t = spot * (1 + annual_carry) ** (t/periods_per_year)
   Lets a user stress-test "what if the market is in X% contango/backwardation?".
2. Real futures curve (EIA NYMEX WTI contracts RCLC1-4) via data.loader — front
   contracts are real; the tail is extrapolated with the front log-slope.
"""

from __future__ import annotations

import numpy as np

from hedging_assistant.contracts import ForwardCurve

_PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12}


def build_parametric_curve(
    spot: float,
    horizon: int,
    annual_carry: float = 0.0,
    frequency: str = "M",
    source: str = "parametric",
) -> ForwardCurve:
    """
    Build a smooth forward curve from a single annualized carry rate.

    annual_carry > 0  -> contango      (forward rises above spot)
    annual_carry < 0  -> backwardation (forward falls below spot)
    annual_carry == 0 -> flat curve at spot (zero basis)

    F_t = spot * (1 + annual_carry) ** (t / periods_per_year),  t = 1..horizon
    """
    if spot <= 0:
        raise ValueError("spot must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    ppy = _PERIODS_PER_YEAR.get(frequency.upper(), 12)
    t = np.arange(1, horizon + 1, dtype=float)
    prices = float(spot) * (1.0 + annual_carry) ** (t / ppy)

    return ForwardCurve(prices=prices, frequency=frequency.upper(), source=source)


def build_curve_from_futures(
    front_contracts: "list[float] | np.ndarray",
    spot: float,
    horizon: int,
    frequency: str = "M",
) -> ForwardCurve:
    """
    Build an H-period curve from a handful of real front futures contracts
    (e.g. EIA RCLC1-4), extrapolating the tail with the front log-slope.

    front_contracts: observed futures prices for periods 1..k (k < horizon ok).
    """
    fut = np.asarray(front_contracts, dtype=float)
    fut = fut[fut > 0]
    if len(fut) == 0:
        # No real quotes -> flat curve at spot.
        return build_parametric_curve(spot, horizon, 0.0, frequency, source="spot-fallback")

    k = len(fut)
    if k >= horizon:
        return ForwardCurve(prices=fut[:horizon], frequency=frequency.upper(), source="eia-futures")

    # Extrapolate months k+1..horizon using the average log-slope of the front.
    if k >= 2:
        slope = float(np.mean(np.diff(np.log(fut))))
    else:
        slope = 0.0
    tail_idx = np.arange(1, horizon - k + 1, dtype=float)
    tail = fut[-1] * np.exp(slope * tail_idx)
    prices = np.concatenate([fut, tail])

    return ForwardCurve(prices=prices, frequency=frequency.upper(), source="eia-futures+extrap")


def implied_annual_carry(spot: float, forward_curve: ForwardCurve) -> float:
    """Back out the average annualized carry implied by a curve (for reporting)."""
    ppy = _PERIODS_PER_YEAR.get(forward_curve.frequency.upper(), 12)
    fwd = np.asarray(forward_curve.prices, dtype=float)
    if spot <= 0 or len(fwd) == 0:
        return 0.0
    # geometric average step, annualized
    log_slope = float(np.mean(np.diff(np.log(np.concatenate([[spot], fwd])))))
    return float(np.exp(log_slope * ppy) - 1.0)
