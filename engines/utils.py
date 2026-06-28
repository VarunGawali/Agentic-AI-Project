"""
Shared utilities for the hedging engine modules.
"""

from __future__ import annotations

import numpy as np


def resolve_forward_curve(
    forward_price,
    horizon: int,
) -> np.ndarray:
    """
    Convert scalar forward price or forward-curve-like object into a 1-D array.

    Supported inputs:
        - scalar float
        - list / np.ndarray of shape (horizon,)
        - object with .prices attribute
    """

    if np.isscalar(forward_price):
        if float(forward_price) <= 0:
            raise ValueError(f"forward_price must be positive; got {forward_price}")

        return np.full(horizon, float(forward_price), dtype=float)

    if hasattr(forward_price, "prices"):
        fwd = np.asarray(forward_price.prices, dtype=float)
    else:
        fwd = np.asarray(forward_price, dtype=float)

    if fwd.ndim != 1:
        raise ValueError("forward curve must be a 1D array")

    if len(fwd) != horizon:
        raise ValueError(
            f"forward curve length {len(fwd)} does not match horizon {horizon}"
        )

    if np.any(fwd <= 0):
        raise ValueError("forward curve prices must all be positive")

    return fwd
