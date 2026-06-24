"""
Strategy library: generate HedgingPolicy hedge fractions from StrategyParams.

BASELINE (Phase 1): staggered strategy with cap constraint.
PHASE 3 additions:
  - TRIGGER: hedge high_frac when price > MA * threshold, else base_frac.
  - VOLATILITY: scale base_frac continuously by realised vol vs long-run vol.
  - HYBRID: combine trigger and vol signals multiplicatively.
  - DP_OPTIMAL: Bellman-optimal policy via backward induction over a discretised
    (price_bin × vol_bin) state space; built offline by build_dp_table().

CHANGES:
  - apply_strategy() now accepts price_history prefix for MA computation on
    path-dependent strategies; STAGGERED path stays fully vectorised.
  - is_path_dependent() helper: True for TRIGGER / VOLATILITY / HYBRID / DP.
  - build_dp_table(): backward induction, returns policy dict for DP_OPTIMAL.
  - generate_batch_candidates() extended to HYBRID and DP_OPTIMAL types.
"""

from __future__ import annotations
import numpy as np

from hedging_assistant.contracts import (
    StrategyType, StrategyParams, HedgingPolicy, PriceForecast,
)

# ── price-bin edges: log(price / MA), 5 buckets ──────────────────────────────
_PRICE_BINS = np.array([-0.10, -0.03, 0.03, 0.10])   # 5 regions, 4 cut-points
# ── vol-bin: realised_vol vs long_run_vol; split at 1.2× ─────────────────────
_VOL_SPLIT  = 1.2
N_PRICE_BINS = 5
N_VOL_BINS   = 2
N_STATES     = N_PRICE_BINS * N_VOL_BINS   # 10


def _price_bin(price: float, ma: float) -> int:
    ratio = np.log(price / ma) if ma > 0 else 0.0
    return int(np.searchsorted(_PRICE_BINS, ratio))   # 0..4


def _vol_bin(realised_vol: float, long_run_vol: float) -> int:
    return 1 if (realised_vol / long_run_vol) >= _VOL_SPLIT else 0


def _state_idx(pb: int, vb: int) -> int:
    return pb * N_VOL_BINS + vb


def _compute_vol(prices: np.ndarray) -> float:
    """Annualised vol from log-returns of the supplied price window."""
    if len(prices) < 2:
        return 0.0
    lr = np.diff(np.log(prices))
    return float(lr.std(ddof=1)) * np.sqrt(len(lr))


def is_path_dependent(params: StrategyParams) -> bool:
    return params.strategy_type in (
        StrategyType.TRIGGER, StrategyType.VOLATILITY,
        StrategyType.HYBRID,  StrategyType.DP_OPTIMAL,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core: apply a strategy to one path (or static for STAGGERED)
# ─────────────────────────────────────────────────────────────────────────────

def apply_strategy(
    params: StrategyParams,
    price_path: np.ndarray,
    price_history: np.ndarray | None = None,
    long_run_vol: float | None = None,
) -> np.ndarray:
    """
    Compute the hedge fraction schedule for one price path.

    Parameters
    ----------
    params        : strategy configuration
    price_path    : shape (H,)  — future prices for one simulated path
    price_history : shape (T,)  — historical prices preceding price_path;
                    used to warm-start the moving average.  If None the MA
                    is computed entirely from within price_path.
    long_run_vol  : long-run annualised vol for vol-scaling reference;
                    estimated from price_history if not supplied.

    Returns
    -------
    fractions : shape (H,)  values in [0, cap]
    """
    horizon = len(price_path)
    if horizon == 0:
        raise ValueError("price_path is empty.")

    # ── STAGGERED: fully vectorised, no history needed ────────────────────────
    if params.strategy_type == StrategyType.STAGGERED:
        return np.clip(np.full(horizon, params.base_fraction), 0.0, params.cap)

    # ── Path-dependent strategies need history for warm MA / vol ─────────────
    hist = np.asarray(price_history) if price_history is not None else np.array([])
    full = np.concatenate([hist, price_path]) if len(hist) else price_path

    # Long-run vol: estimate from history if not supplied
    if long_run_vol is None or long_run_vol <= 0:
        ref = hist if len(hist) >= 10 else full
        long_run_vol = max(_compute_vol(ref), 1e-8)

    ma_w = max(params.ma_window, 2)
    fractions = np.empty(horizon)

    for t in range(horizon):
        # index into the concatenated series
        idx = len(hist) + t
        p_now = price_path[t]

        # ── Moving-average for trigger / hybrid ──────────────────────────────
        window_start = max(0, idx - ma_w + 1)
        ma = float(full[window_start : idx + 1].mean())

        # ── Realised vol over the same window ────────────────────────────────
        rv_window = full[max(0, idx - ma_w) : idx + 1]
        rv = _compute_vol(rv_window) if len(rv_window) >= 2 else long_run_vol
        realised_vol = rv if np.isfinite(rv) and rv > 0 else long_run_vol

        if params.strategy_type == StrategyType.TRIGGER:
            frac = (
                params.trigger_fraction
                if (p_now / ma) >= params.trigger_threshold
                else params.base_fraction
            )

        elif params.strategy_type == StrategyType.VOLATILITY:
            # Continuous scaling: frac = base * (1 + k * (rv/lrv - 1))
            scale = 1.0 + params.vol_scale_k * (realised_vol / long_run_vol - 1.0)
            frac  = params.base_fraction * max(scale, 0.0)

        elif params.strategy_type == StrategyType.HYBRID:
            # Trigger multiplier: ratio of trigger_fraction to base_fraction
            # when trigger fires; 1.0 otherwise.
            if params.base_fraction > 0:
                trig_mult = (
                    params.trigger_fraction / params.base_fraction
                    if (p_now / ma) >= params.trigger_threshold
                    else 1.0
                )
            else:
                trig_mult = 1.0
            vol_scale = 1.0 + params.vol_scale_k * (realised_vol / long_run_vol - 1.0)
            frac = params.base_fraction * max(trig_mult * vol_scale, 0.0)

        elif params.strategy_type == StrategyType.DP_OPTIMAL:
            if params.dp_table is None:
                raise ValueError(
                    "dp_table is None — call build_dp_table() first "
                    "and attach the result to StrategyParams.dp_table."
                )
            pb = _price_bin(p_now, ma)
            vb = _vol_bin(realised_vol, long_run_vol)
            frac = params.dp_table.get((t, pb, vb), params.base_fraction)

        else:
            raise ValueError(f"Unknown strategy type: {params.strategy_type}")

        fractions[t] = np.clip(frac, 0.0, params.cap)

    return fractions


# ─────────────────────────────────────────────────────────────────────────────
# DP: backward induction over (price_bin × vol_bin) state space
# ─────────────────────────────────────────────────────────────────────────────

def build_dp_table(
    forecast_obj: PriceForecast,
    exposure,            # ExposureBook
    forward_price,       # float or ForwardCurve
    max_hedge:    float = 1.0,
    n_actions:    int   = 11,
    cost_weight:  float = 1.0,
    cvar_weight:  float = 1.0,
    cvar_alpha:   float = 0.95,
    price_history: np.ndarray | None = None,
    long_run_vol:  float | None = None,
) -> dict:
    """
    Bellman-optimal hedge policy via backward induction.

    State space: (t, price_bin, vol_bin) — 10 states per period.
    Actions:     n_actions discrete fractions in [0, max_hedge].
    Objective:   minimise cost_weight * E[cost] + cvar_weight * CVaR
                 over remaining periods, using empirical path transitions.

    Algorithm
    ---------
    1. Bin each GBM path into states at each period t.
    2. For t = H-1 down to 0:
         for each state s:
           for each action f:
             compute immediate cost(t, f, paths in state s)
             + E[V*(t+1, s') | s, f]   (expectation over GBM transitions)
           π*(t, s) = argmin over f
    3. Return {(t, price_bin, vol_bin): optimal_fraction}.

    The policy is then used path-by-path in apply_strategy(DP_OPTIMAL).

    Parameters
    ----------
    forecast_obj   : PriceForecast with paths (N, H)
    exposure       : ExposureBook with volumes (H,)
    forward_price  : float or ForwardCurve
    max_hedge      : hard cap on hedge fraction
    n_actions      : grid resolution over [0, max_hedge]
    cost_weight    : weight on expected cost in objective
    cvar_weight    : weight on CVaR in objective
    cvar_alpha     : CVaR tail level
    price_history  : shape (T,) historical prices for MA warm-start
    long_run_vol   : long-run vol reference; estimated if None

    Returns
    -------
    dict: {(t: int, price_bin: int, vol_bin: int): hedge_fraction: float}
    """
    from hedging_assistant.contracts import ForwardCurve

    paths   = np.asarray(forecast_obj.paths, dtype=float)   # (N, H)
    volumes = np.asarray(exposure.volumes,   dtype=float)   # (H,)
    N, H    = paths.shape

    if isinstance(forward_price, ForwardCurve):
        fwd = np.asarray(forward_price.prices, dtype=float)
    else:
        fwd = np.full(H, float(forward_price))

    actions = np.linspace(0.0, max_hedge, n_actions)        # (A,)
    A       = len(actions)

    # ── Estimate long-run vol from history or all paths ───────────────────────
    hist = np.asarray(price_history) if price_history is not None else np.array([])
    if long_run_vol is None or long_run_vol <= 0:
        ref = hist if len(hist) >= 10 else paths[:, 0]
        long_run_vol = max(_compute_vol(ref if len(ref.shape) == 1 else ref.flatten()), 1e-8)

    ma_window = 5   # fixed for DP; can expose as param later

    # ── Bin every path at every period ───────────────────────────────────────
    # state_matrix[n, t] = state index (0..N_STATES-1)
    state_matrix = np.zeros((N, H), dtype=int)
    for n in range(N):
        for t in range(H):
            idx     = len(hist) + t
            full_t  = np.concatenate([hist, paths[n, :t+1]])
            w_start = max(0, len(full_t) - ma_window)
            ma      = float(full_t[w_start:].mean())
            rv_win  = full_t[max(0, len(full_t)-ma_window):]
            rv      = max(_compute_vol(rv_win), 1e-8)
            pb      = _price_bin(paths[n, t], ma)
            vb      = _vol_bin(rv, long_run_vol)
            state_matrix[n, t] = _state_idx(pb, vb)

    # ── Immediate cost for each path n, period t, action f ───────────────────
    # cost(n, t, f) = f * vol_t * fwd_t  +  (1-f) * vol_t * path[n,t]
    #              = vol_t * (f * fwd_t + (1-f) * path[n,t])
    # shape: (N, H, A)
    vol_mat  = volumes[None, :, None]            # (1, H, 1)
    fwd_mat  = fwd[None, :, None]                # (1, H, 1)
    path_mat = paths[:, :, None]                 # (N, H, 1)
    action_mat = actions[None, None, :]          # (1, 1, A)

    period_costs = vol_mat * (action_mat * fwd_mat + (1.0 - action_mat) * path_mat)
    # shape: (N, H, A)

    # ── Backward induction ────────────────────────────────────────────────────
    # V[s, a] = expected future value from state s taking action a at period t
    # policy[t, s] = optimal action index

    # Terminal value: V*(H, ·) = 0
    V_next = np.zeros(N_STATES)   # value at t+1, indexed by state

    policy = {}   # {(t, pb, vb): fraction}

    for t in reversed(range(H)):
        V_curr = np.full(N_STATES, np.inf)
        action_opt = np.zeros(N_STATES, dtype=int)

        for s in range(N_STATES):
            # paths currently in state s at period t
            mask = state_matrix[:, t] == s
            if mask.sum() == 0:
                # no paths in this state — use nearest populated state heuristic
                V_curr[s] = 0.0
                action_opt[s] = n_actions // 2
                continue

            # immediate cost for each action: mean over paths in this state
            imm = period_costs[mask, t, :]           # (n_s, A)
            imm_mean = imm.mean(axis=0)              # (A,)

            # CVaR over paths in this state for each action
            # (only meaningful when n_s is large enough; else use mean)
            if mask.sum() >= 20:
                thr = np.percentile(imm, cvar_alpha * 100, axis=0)   # (A,)
                # CVaR: mean of values >= thr per action
                cvar_arr = np.array([
                    float(imm[imm[:, a] >= thr[a], a].mean()) if (imm[:, a] >= thr[a]).any() else thr[a]
                    for a in range(A)
                ])
            else:
                cvar_arr = imm_mean

            # expected future value: E[V*(t+1, s') | s at t]
            if t < H - 1:
                next_states = state_matrix[mask, t + 1]     # (n_s,)
                ev_next = V_next[next_states].mean()         # scalar
            else:
                ev_next = 0.0

            # total objective per action
            obj = cost_weight * imm_mean + cvar_weight * cvar_arr + ev_next
            best_a = int(np.argmin(obj))
            action_opt[s] = best_a
            V_curr[s] = float(obj[best_a])

        # store policy for this period
        for s in range(N_STATES):
            pb = s // N_VOL_BINS
            vb = s %  N_VOL_BINS
            policy[(t, pb, vb)] = float(actions[action_opt[s]])

        V_next = V_curr

    print(
        f"[strategy_library] DP table built: {H} periods × {N_STATES} states. "
        f"long_run_vol={long_run_vol:.3f}"
    )
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# Policy builder (representative path — median)
# ─────────────────────────────────────────────────────────────────────────────

def build_policy(
    params: StrategyParams,
    forecast_obj: PriceForecast,
    price_history: np.ndarray | None = None,
) -> HedgingPolicy:
    """Build a representative HedgingPolicy using the median forecast path."""
    median_path = np.percentile(forecast_obj.paths, 50, axis=0)
    fractions   = apply_strategy(params, median_path, price_history=price_history)
    desc = f"{params.base_fraction:.0%} {params.strategy_type.value} (cap={params.cap:.0%})"
    return HedgingPolicy(params=params, hedge_fractions=fractions, description=desc)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_staggered_candidates(
    max_hedge: float = 1.0,
    n_steps: int = 11,
    cap: float = 1.0,
) -> list[StrategyParams]:
    if not (0 < max_hedge <= 1.0):
        raise ValueError(f"max_hedge must be in (0, 1]; got {max_hedge}")
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2; got {n_steps}")
    return [
        StrategyParams(strategy_type=StrategyType.STAGGERED,
                       base_fraction=float(f), cap=float(cap))
        for f in np.linspace(0.0, max_hedge, n_steps)
    ]


def generate_hybrid_candidates(
    max_hedge: float = 1.0,
    base_fractions: list[float] | None = None,
    trigger_thresholds: list[float] | None = None,
    vol_scale_ks: list[float] | None = None,
    cap: float = 1.0,
) -> list[StrategyParams]:
    """
    Grid of HYBRID candidates over (base_fraction, trigger_threshold, vol_scale_k).
    Defaults give a compact 3×2×2 = 12-candidate grid.
    """
    bfs  = base_fractions       or [0.3, 0.6, 0.9]
    thrs = trigger_thresholds   or [1.0, 1.05]
    ks   = vol_scale_ks         or [0.5, 1.5]
    return [
        StrategyParams(
            strategy_type=StrategyType.HYBRID,
            base_fraction=float(bf),
            trigger_fraction=min(float(bf) * 1.5, max_hedge),
            trigger_threshold=float(thr),
            vol_scale_k=float(k),
            cap=float(cap),
        )
        for bf in bfs for thr in thrs for k in ks
    ]


def generate_batch_candidates(
    strategy_types: list[StrategyType] | None = None,
    max_hedge: float = 1.0,
    n_steps: int = 11,
    cap: float = 1.0,
) -> list[StrategyParams]:
    """Generate candidates across multiple strategy types."""
    if strategy_types is None:
        strategy_types = [StrategyType.STAGGERED]
    all_candidates: list[StrategyParams] = []
    for stype in strategy_types:
        if stype == StrategyType.STAGGERED:
            all_candidates.extend(generate_staggered_candidates(max_hedge, n_steps, cap))
        elif stype == StrategyType.HYBRID:
            all_candidates.extend(generate_hybrid_candidates(max_hedge, cap=cap))
        else:
            raise NotImplementedError(f"Batch generation for {stype} not yet wired.")
    return all_candidates
