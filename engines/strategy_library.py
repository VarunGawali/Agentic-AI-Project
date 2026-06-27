"""
Strategy library: generate hedge fractions from StrategyParams.

Phase 1:
    - STAGGERED hedge strategy

Phase 3:
    - TRIGGER strategy
    - VOLATILITY strategy
    - HYBRID strategy
    - DP_OPTIMAL strategy using offline dynamic-programming table

Notes:
    - STAGGERED is path-independent and can be vectorized by simulator.
    - TRIGGER / VOLATILITY / HYBRID / DP_OPTIMAL are path-dependent.
"""

from __future__ import annotations

import numpy as np

from contracts import (
    StrategyType,
    StrategyParams,
    HedgingPolicy,
    PriceForecast,
)


# ---------------------------------------------------------------------------
# DP state-space constants
# ---------------------------------------------------------------------------

_PRICE_BINS = np.array([-0.10, -0.03, 0.03, 0.10])
_VOL_SPLIT = 1.2

N_PRICE_BINS = 5
N_VOL_BINS = 2
N_STATES = N_PRICE_BINS * N_VOL_BINS


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _price_bin(
    price: float,
    ma: float,
) -> int:
    """
    Convert price relative to moving average into one of 5 bins.
    """

    if ma <= 0 or not np.isfinite(ma):
        ratio = 0.0
    else:
        ratio = np.log(price / ma)

    return int(np.searchsorted(_PRICE_BINS, ratio))


def _vol_bin(
    realised_vol: float,
    long_run_vol: float,
) -> int:
    """
    Convert realised volatility into low-vol/high-vol bin.
    """

    if long_run_vol <= 0 or not np.isfinite(long_run_vol):
        return 0

    return 1 if (realised_vol / long_run_vol) >= _VOL_SPLIT else 0


def _state_idx(
    price_bin: int,
    vol_bin: int,
) -> int:
    """
    Convert price-bin and vol-bin to flattened state index.
    """

    return price_bin * N_VOL_BINS + vol_bin


def _compute_vol(
    prices: np.ndarray,
) -> float:
    """
    Compute annualized volatility from log returns of supplied price window.

    Note:
        Since the window can be monthly/daily depending on context, this is a
        local realized-vol proxy, not a fully calendar-annualized measure.
    """

    prices = np.asarray(prices, dtype=float)

    if len(prices) < 2:
        return 0.0

    prices = prices[prices > 0]

    if len(prices) < 2:
        return 0.0

    log_returns = np.diff(np.log(prices))

    if len(log_returns) < 2:
        return 0.0

    vol = float(log_returns.std(ddof=1)) * np.sqrt(len(log_returns))

    if not np.isfinite(vol):
        return 0.0

    return vol


def is_path_dependent(params: StrategyParams) -> bool:
    return params.strategy_type in {
        StrategyType.TRIGGER,
        StrategyType.VOLATILITY,
        StrategyType.HYBRID,
        StrategyType.DP_OPTIMAL,
    }



# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_params(
    params: StrategyParams,
) -> None:
    """
    Validate common strategy parameters.
    """

    if not 0 <= params.base_fraction <= 1:
        raise ValueError("base_fraction must be between 0 and 1")

    if not 0 <= params.cap <= 1:
        raise ValueError("cap must be between 0 and 1")

    if hasattr(params, "trigger_fraction"):
        if not 0 <= params.trigger_fraction <= 1:
            raise ValueError("trigger_fraction must be between 0 and 1")

    if hasattr(params, "trigger_threshold"):
        if params.trigger_threshold <= 0:
            raise ValueError("trigger_threshold must be positive")

    if hasattr(params, "ma_window"):
        if params.ma_window < 2:
            raise ValueError("ma_window must be at least 2")


def _prepare_price_path(
    price_path: np.ndarray,
) -> np.ndarray:
    """
    Convert and validate one simulated price path.
    """

    price_path = np.asarray(price_path, dtype=float)

    if price_path.ndim != 1:
        raise ValueError("price_path must be a 1D array")

    if len(price_path) == 0:
        raise ValueError("price_path cannot be empty")

    if np.any(price_path <= 0):
        raise ValueError("price_path must contain only positive prices")

    return price_path


# ---------------------------------------------------------------------------
# Core strategy application
# ---------------------------------------------------------------------------

def apply_strategy(
    params: StrategyParams,
    price_path: np.ndarray,
    price_history: np.ndarray | None = None,
    long_run_vol: float | None = None,
    history_vol: float | None = None,
) -> np.ndarray:
    """
    Compute hedge fraction schedule for one simulated price path.

    Parameters:
        params:
            Strategy configuration.

        price_path:
            Future simulated price path, shape = (horizon,).

        price_history:
            Historical prices before price_path.
            Used to warm-start moving average and volatility calculations.

        long_run_vol:
            Long-run volatility reference for VOLATILITY/HYBRID/DP strategies.

        history_vol:
            Backward-compatible alias from older code.
            If long_run_vol is not provided, history_vol is used.

    Returns:
        Hedge fractions, shape = (horizon,), clipped to [0, cap].
    """

    price_path = _prepare_price_path(price_path)
    _validate_params(params)

    horizon = len(price_path)

    if long_run_vol is None and history_vol is not None:
        long_run_vol = history_vol

    # -----------------------------------------------------------------------
    # STAGGERED: path-independent, fully vectorizable
    # -----------------------------------------------------------------------

    if params.strategy_type == StrategyType.STAGGERED:
        hedge_fraction = min(params.base_fraction, params.cap)
        return np.full(horizon, hedge_fraction, dtype=float)
    
    if params.strategy_type == StrategyType.CVAR_LP:
        if params.fixed_fractions is None:
            raise ValueError(
                "fixed_fractions is required for CVAR_LP strategy."
            )

        fixed = np.asarray(params.fixed_fractions, dtype=float)

        if len(fixed) != horizon:
            raise ValueError(
                f"CVAR_LP fixed_fractions length {len(fixed)} does not match "
                f"forecast horizon {horizon}"
            )

        return np.clip(fixed, 0.0, params.cap)

    # -----------------------------------------------------------------------
    # Path-dependent strategies
    # -----------------------------------------------------------------------

    hist = (
        np.asarray(price_history, dtype=float)
        if price_history is not None
        else np.array([], dtype=float)
    )

    hist = hist[hist > 0]

    if len(hist):
        full = np.concatenate([hist, price_path])
    else:
        full = price_path

    if long_run_vol is None or long_run_vol <= 0 or not np.isfinite(long_run_vol):
        reference_prices = hist if len(hist) >= 10 else full
        long_run_vol = max(_compute_vol(reference_prices), 1e-8)

    ma_window = max(int(getattr(params, "ma_window", 5)), 2)

    fractions = np.empty(horizon, dtype=float)

    for t in range(horizon):
        idx = len(hist) + t
        p_now = float(price_path[t])

        window_start = max(0, idx - ma_window + 1)
        ma = float(full[window_start: idx + 1].mean())

        rv_window = full[max(0, idx - ma_window): idx + 1]
        realised_vol = _compute_vol(rv_window)

        if not np.isfinite(realised_vol) or realised_vol <= 0:
            realised_vol = long_run_vol

        # -------------------------------------------------------------------
        # TRIGGER
        # -------------------------------------------------------------------

        if params.strategy_type == StrategyType.TRIGGER:
            if (p_now / ma) >= params.trigger_threshold:
                frac = params.trigger_fraction
            else:
                frac = params.base_fraction

        # -------------------------------------------------------------------
        # VOLATILITY
        # -------------------------------------------------------------------

        elif params.strategy_type == StrategyType.VOLATILITY:
            scale = 1.0 + params.vol_scale_k * (
                realised_vol / long_run_vol - 1.0
            )

            frac = params.base_fraction * max(scale, 0.0)

        # -------------------------------------------------------------------
        # HYBRID
        # -------------------------------------------------------------------

        elif params.strategy_type == StrategyType.HYBRID:
            if params.base_fraction > 0:
                if (p_now / ma) >= params.trigger_threshold:
                    trigger_multiplier = (
                        params.trigger_fraction / params.base_fraction
                    )
                else:
                    trigger_multiplier = 1.0
            else:
                trigger_multiplier = 1.0

            vol_scale = 1.0 + params.vol_scale_k * (
                realised_vol / long_run_vol - 1.0
            )

            frac = params.base_fraction * max(
                trigger_multiplier * vol_scale,
                0.0,
            )

        # -------------------------------------------------------------------
        # DP_OPTIMAL
        # -------------------------------------------------------------------

        elif params.strategy_type == StrategyType.DP_OPTIMAL:
            if params.dp_table is None:
                raise ValueError(
                    "dp_table is None. Call build_dp_table() first and attach "
                    "the result to StrategyParams.dp_table."
                )

            pb = _price_bin(p_now, ma)
            vb = _vol_bin(realised_vol, long_run_vol)

            frac = params.dp_table.get(
                (t, pb, vb),
                params.base_fraction,
            )

        else:
            raise ValueError(f"Unknown strategy type: {params.strategy_type}")

        fractions[t] = np.clip(frac, 0.0, params.cap)

    return fractions


# ---------------------------------------------------------------------------
# Forward price helper for DP
# ---------------------------------------------------------------------------

def _resolve_forward_curve(
    forward_price,
    horizon: int,
) -> np.ndarray:
    """
    Resolve scalar/list/ForwardCurve-like object into forward curve array.

    Supported:
        - scalar float
        - list/np.ndarray shape (horizon,)
        - object with .prices
    """

    if np.isscalar(forward_price):
        if float(forward_price) <= 0:
            raise ValueError("forward_price must be positive")

        return np.full(horizon, float(forward_price), dtype=float)

    if hasattr(forward_price, "prices"):
        fwd = np.asarray(forward_price.prices, dtype=float)
    else:
        fwd = np.asarray(forward_price, dtype=float)

    if fwd.ndim != 1:
        raise ValueError("forward curve must be 1D")

    if len(fwd) != horizon:
        raise ValueError(
            f"forward curve length {len(fwd)} does not match horizon {horizon}"
        )

    if np.any(fwd <= 0):
        raise ValueError("forward curve prices must be positive")

    return fwd


# ---------------------------------------------------------------------------
# DP table builder
# ---------------------------------------------------------------------------

def build_dp_table(
    forecast_obj: PriceForecast,
    exposure,
    forward_price,
    max_hedge: float = 1.0,
    n_actions: int = 11,
    cost_weight: float = 1.0,
    cvar_weight: float = 1.0,
    cvar_alpha: float = 0.95,
    price_history: np.ndarray | None = None,
    long_run_vol: float | None = None,
) -> dict:
    """
    Build Bellman-style DP hedge policy over a discretized state space.

    State:
        (period t, price_bin, vol_bin)

    Action:
        Hedge fraction from discrete grid in [0, max_hedge]

    Objective:
        cost_weight * expected immediate cost
        + cvar_weight * immediate CVaR
        + expected next-state value

    Returns:
        dict mapping:
            (t, price_bin, vol_bin) -> optimal hedge fraction
    """

    paths = np.asarray(forecast_obj.paths, dtype=float)
    volumes = np.asarray(exposure.volumes, dtype=float)

    if paths.ndim != 2:
        raise ValueError("forecast_obj.paths must be 2D")

    n_paths, horizon = paths.shape

    if len(volumes) != horizon:
        raise ValueError(
            f"exposure length {len(volumes)} does not match horizon {horizon}"
        )

    if not 0 < max_hedge <= 1:
        raise ValueError("max_hedge must be in (0, 1]")

    if n_actions < 2:
        raise ValueError("n_actions must be at least 2")

    if not 0 < cvar_alpha < 1:
        raise ValueError("cvar_alpha must be between 0 and 1")

    fwd = _resolve_forward_curve(forward_price, horizon)

    actions = np.linspace(0.0, max_hedge, n_actions)
    n_actions_actual = len(actions)

    hist = (
        np.asarray(price_history, dtype=float)
        if price_history is not None
        else np.array([], dtype=float)
    )

    hist = hist[hist > 0]

    if long_run_vol is None or long_run_vol <= 0 or not np.isfinite(long_run_vol):
        reference = hist if len(hist) >= 10 else paths.flatten()
        long_run_vol = max(_compute_vol(reference), 1e-8)

    ma_window = 5

    state_matrix = np.zeros((n_paths, horizon), dtype=int)

    for n in range(n_paths):
        for t in range(horizon):
            if len(hist):
                full_t = np.concatenate([hist, paths[n, : t + 1]])
            else:
                full_t = paths[n, : t + 1]

            ma = float(full_t[max(0, len(full_t) - ma_window):].mean())

            rv_window = full_t[max(0, len(full_t) - ma_window):]
            realised_vol = max(_compute_vol(rv_window), 1e-8)

            pb = _price_bin(paths[n, t], ma)
            vb = _vol_bin(realised_vol, long_run_vol)

            state_matrix[n, t] = _state_idx(pb, vb)

    vol_mat = volumes[None, :, None]
    fwd_mat = fwd[None, :, None]
    path_mat = paths[:, :, None]
    action_mat = actions[None, None, :]

    period_costs = vol_mat * (
        action_mat * fwd_mat
        + (1.0 - action_mat) * path_mat
    )

    value_next = np.zeros(N_STATES, dtype=float)
    policy: dict[tuple[int, int, int], float] = {}

    for t in reversed(range(horizon)):
        value_curr = np.full(N_STATES, np.inf, dtype=float)
        action_opt = np.zeros(N_STATES, dtype=int)

        for state in range(N_STATES):
            mask = state_matrix[:, t] == state
            n_state = int(mask.sum())

            if n_state == 0:
                value_curr[state] = 0.0
                action_opt[state] = n_actions_actual // 2
                continue

            immediate = period_costs[mask, t, :]
            immediate_mean = immediate.mean(axis=0)

            if n_state >= 20:
                thresholds = np.percentile(
                    immediate,
                    cvar_alpha * 100,
                    axis=0,
                )

                cvar_values = np.array(
                    [
                        float(
                            immediate[immediate[:, a] >= thresholds[a], a].mean()
                        )
                        if np.any(immediate[:, a] >= thresholds[a])
                        else float(thresholds[a])
                        for a in range(n_actions_actual)
                    ],
                    dtype=float,
                )
            else:
                cvar_values = immediate_mean

            if t < horizon - 1:
                next_states = state_matrix[mask, t + 1]
                expected_next_value = float(value_next[next_states].mean())
            else:
                expected_next_value = 0.0

            objective = (
                cost_weight * immediate_mean
                + cvar_weight * cvar_values
                + expected_next_value
            )

            best_action = int(np.argmin(objective))
            action_opt[state] = best_action
            value_curr[state] = float(objective[best_action])

        for state in range(N_STATES):
            pb = state // N_VOL_BINS
            vb = state % N_VOL_BINS
            policy[(t, pb, vb)] = float(actions[action_opt[state]])

        value_next = value_curr

    print(
        f"[strategy_library] DP table built: "
        f"{horizon} periods x {N_STATES} states. "
        f"long_run_vol={long_run_vol:.4f}"
    )

    return policy


# ---------------------------------------------------------------------------
# Policy builder
# ---------------------------------------------------------------------------

def build_policy(
    params: StrategyParams,
    forecast_obj: PriceForecast,
    price_history: np.ndarray | None = None,
) -> HedgingPolicy:
    """
    Build a representative HedgingPolicy using the median forecast path.
    """

    paths = np.asarray(forecast_obj.paths, dtype=float)

    if paths.ndim != 2:
        raise ValueError("forecast_obj.paths must be 2D: (n_paths, horizon)")

    if paths.shape[1] == 0:
        raise ValueError("forecast horizon cannot be zero")

    median_path = np.percentile(paths, 50, axis=0)

    hedge_fractions = apply_strategy(
        params=params,
        price_path=median_path,
        price_history=price_history,
    )

    if params.strategy_type == StrategyType.CVAR_LP:
        description = f"CVaR-LP optimized hedge schedule with cap {params.cap:.0%}"
    else:
        description = (
            f"{params.base_fraction:.0%} {params.strategy_type.value} hedge "
            f"with cap {params.cap:.0%}"
        )

    return HedgingPolicy(
        params=params,
        hedge_fractions=hedge_fractions,
        description=description,
    )


# ---------------------------------------------------------------------------
# Candidate generators
# ---------------------------------------------------------------------------

def generate_staggered_candidates(
    fractions: list[float] | None = None,
    max_hedge: float = 1.0,
    n_steps: int = 5,
    cap: float = 1.0,
) -> list:
    """
    Generate staggered hedge candidates.

    Supports:
        explicit fractions=[...]
        or grid max_hedge/n_steps.
    """

    if not 0 <= max_hedge <= 1:
        raise ValueError("max_hedge must be between 0 and 1")

    if not 0 <= cap <= 1:
        raise ValueError("cap must be between 0 and 1")

    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")

    effective_max = min(max_hedge, cap)

    if fractions is None:
        fractions_array = np.linspace(0.0, effective_max, n_steps)
    else:
        fractions_array = np.asarray(fractions, dtype=float)

    candidates: list[StrategyParams] = []
    seen: set[float] = set()

    for frac in fractions_array:
        if not 0 <= frac <= 1:
            raise ValueError("all hedge fractions must be between 0 and 1")

        effective_frac = min(float(frac), effective_max)
        rounded_frac = round(effective_frac, 10)

        if rounded_frac in seen:
            continue

        seen.add(rounded_frac)

        candidates.append(
            StrategyParams(
                strategy_type=StrategyType.STAGGERED,
                base_fraction=effective_frac,
                cap=cap,
            )
        )

    return candidates


def generate_trigger_candidates(
    max_hedge: float = 1.0,
    base_fractions: list[float] | None = None,
    trigger_thresholds: list[float] | None = None,
    trigger_multipliers: list[float] | None = None,
    ma_window: int = 5,
    cap: float = 1.0,
) -> list:
    """
    Generate trigger-based candidates.
    """

    base_fractions = base_fractions or [0.25, 0.50, 0.75]
    trigger_thresholds = trigger_thresholds or [1.00, 1.05, 1.10]
    trigger_multipliers = trigger_multipliers or [1.25, 1.50]

    candidates: list[StrategyParams] = []

    effective_cap = min(max_hedge, cap)

    for base_fraction in base_fractions:
        for threshold in trigger_thresholds:
            for multiplier in trigger_multipliers:
                trigger_fraction = min(
                    float(base_fraction) * float(multiplier),
                    effective_cap,
                )

                candidates.append(
                    StrategyParams(
                        strategy_type=StrategyType.TRIGGER,
                        base_fraction=min(float(base_fraction), effective_cap),
                        trigger_fraction=trigger_fraction,
                        trigger_threshold=float(threshold),
                        ma_window=ma_window,
                        cap=effective_cap,
                    )
                )

    return candidates


def generate_volatility_candidates(
    max_hedge: float = 1.0,
    base_fractions: list[float] | None = None,
    vol_scale_ks: list[float] | None = None,
    ma_window: int = 5,
    cap: float = 1.0,
) -> list:
    """
    Generate volatility-scaling candidates.
    """

    base_fractions = base_fractions or [0.25, 0.50, 0.75]
    vol_scale_ks = vol_scale_ks or [0.5, 1.0, 1.5]

    effective_cap = min(max_hedge, cap)

    candidates: list[StrategyParams] = []

    for base_fraction in base_fractions:
        for k in vol_scale_ks:
            candidates.append(
                StrategyParams(
                    strategy_type=StrategyType.VOLATILITY,
                    base_fraction=min(float(base_fraction), effective_cap),
                    vol_scale_k=float(k),
                    ma_window=ma_window,
                    cap=effective_cap,
                )
            )

    return candidates


def generate_hybrid_candidates(
    max_hedge: float = 1.0,
    base_fractions: list[float] | None = None,
    trigger_thresholds: list[float] | None = None,
    vol_scale_ks: list[float] | None = None,
    ma_window: int = 5,
    cap: float = 1.0,
) -> list:
    """
    Generate HYBRID candidates over:
        base_fraction
        trigger_threshold
        vol_scale_k
    """

    base_fractions = base_fractions or [0.30, 0.60, 0.90]
    trigger_thresholds = trigger_thresholds or [1.00, 1.05]
    vol_scale_ks = vol_scale_ks or [0.50, 1.50]

    effective_cap = min(max_hedge, cap)

    candidates: list[StrategyParams] = []

    for base_fraction in base_fractions:
        for threshold in trigger_thresholds:
            for k in vol_scale_ks:
                base = min(float(base_fraction), effective_cap)
                trigger_fraction = min(base * 1.5, effective_cap)

                candidates.append(
                    StrategyParams(
                        strategy_type=StrategyType.HYBRID,
                        base_fraction=base,
                        trigger_fraction=trigger_fraction,
                        trigger_threshold=float(threshold),
                        vol_scale_k=float(k),
                        ma_window=ma_window,
                        cap=effective_cap,
                    )
                )

    return candidates


def generate_batch_candidates(
    strategy_types: list[StrategyType] | None = None,
    max_hedge: float = 1.0,
    n_steps: int = 5,
    cap: float = 1.0,
) -> list:
    """
    Generate candidates across multiple strategy types.
    """

    if strategy_types is None:
        strategy_types = [StrategyType.STAGGERED]

    all_candidates: list[StrategyParams] = []

    for strategy_type in strategy_types:
        if strategy_type == StrategyType.STAGGERED:
            all_candidates.extend(
                generate_staggered_candidates(
                    max_hedge=max_hedge,
                    n_steps=n_steps,
                    cap=cap,
                )
            )

        elif strategy_type == StrategyType.TRIGGER:
            all_candidates.extend(
                generate_trigger_candidates(
                    max_hedge=max_hedge,
                    cap=cap,
                )
            )

        elif strategy_type == StrategyType.VOLATILITY:
            all_candidates.extend(
                generate_volatility_candidates(
                    max_hedge=max_hedge,
                    cap=cap,
                )
            )

        elif strategy_type == StrategyType.HYBRID:
            all_candidates.extend(
                generate_hybrid_candidates(
                    max_hedge=max_hedge,
                    cap=cap,
                )
            )

        elif strategy_type == StrategyType.DP_OPTIMAL:
            raise NotImplementedError(
                "DP_OPTIMAL candidates require a precomputed dp_table. "
                "Build with build_dp_table() and create StrategyParams manually."
            )

        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    return all_candidates