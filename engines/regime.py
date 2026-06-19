"""
engines/regime.py

CHANGES:
  - New file — 2-state HMM regime detector for crude oil vol regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RegimeState:
    current_regime: int          # 0=low-vol, 1=high-vol
    regime_probs: np.ndarray     # shape (2,) posterior probabilities
    mu: np.ndarray               # shape (2,) per-regime drift
    sigma: np.ndarray            # shape (2,) per-regime volatility
    label: str                   # "low-vol" or "high-vol"


def detect_regime(series: pd.Series, n_components: int = 2, seed: int = 42) -> RegimeState:
    """
    Fit a 2-state Gaussian HMM on log-returns and return the current regime.

    States are ordered so that state 0 = low-vol and state 1 = high-vol
    (lower variance state is mapped to regime 0 in the output regardless of
    how hmmlearn internally labels them).
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return RegimeState(
            current_regime=0,
            regime_probs=np.array([1.0, 0.0]),
            mu=np.zeros(2),
            sigma=np.ones(2),
            label="low-vol (HMM unavailable)",
        )

    log_ret = np.diff(np.log(series.values)).reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=100,
        random_state=seed,
    )
    model.fit(log_ret)

    posteriors = model.predict_proba(log_ret)   # shape (T, n_components)
    hidden_states = model.predict(log_ret)      # shape (T,)

    # Extract per-state means and variances from hmmlearn
    raw_means = model.means_.flatten()           # shape (n_components,)
    raw_vars = np.array([model.covars_[i][0, 0] for i in range(n_components)])

    # Map internal states to ordered states: 0=low-vol, 1=high-vol
    # The state with lower variance is "low-vol" → becomes output state 0
    order = np.argsort(raw_vars)   # order[0] = internal idx of low-vol state

    # Reordered mu/sigma in output space (index 0 = low, index 1 = high)
    out_mu = raw_means[order]
    out_sigma = np.sqrt(raw_vars[order])

    # Determine the current (last) internal state and map it to output space
    last_internal_state = int(hidden_states[-1])
    # Which output index corresponds to this internal state?
    internal_to_output = {int(order[i]): i for i in range(n_components)}
    current_regime = internal_to_output[last_internal_state]

    # Reorder the last posterior row to match output ordering
    last_posterior_raw = posteriors[-1]          # shape (n_components,)
    regime_probs = last_posterior_raw[order]     # reorder to output space

    label = "low-vol" if current_regime == 0 else "high-vol"

    return RegimeState(
        current_regime=current_regime,
        regime_probs=regime_probs,
        mu=out_mu,
        sigma=out_sigma,
        label=label,
    )
