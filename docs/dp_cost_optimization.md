# Phase 4: DP-Based Cost Optimization

## Motivation

The current pipeline is a **heuristic search**: it generates ~20 candidate strategies
(staggered, trigger, volatility, hybrid, CVaR-LP), simulates each one, scores them
with a proxy blended metric, and picks the winner. The scorer is an approximation —
it tries to capture what "good" looks like without directly solving for it.

The Phase 4 improvement replaces this with **optimal control via dynamic programming**.
The DP directly minimises the actual procurement cost objective, including CVaR tail
risk, over the full price path distribution. No proxy metric, no heuristic rules, no
candidate grid — one solve, guaranteed-optimal policy.

---

## Objective Function

The cost the buyer faces over horizon H is:

```
total_cost = Σ_t  volume_t × [ f_t × forward_t + (1 - f_t) × spot_t ]
```

where `f_t ∈ [0, max_hedge]` is the hedge fraction chosen at period t.

The full objective (matching the existing 4-factor LP weights):

```
minimise  w_cost × E[total_cost]
        + w_cvar × CVaR_alpha(total_cost)
        + w_opportunity × E[max(total_cost − no_hedge_cost, 0)]
        + w_execution  × execution_cost_per_barrel × Σ_t(f_t × volume_t)
```

CVaR is handled via **Lagrangian relaxation**: the tail risk term decomposes into a
per-scenario penalty so the Bellman update stays a standard expectation:

```
CVaR_alpha ≈ z + 1/((1-alpha)×N) × Σ_i max(cost_i − z, 0)
```

which is just an augmented per-path cost — no state-space expansion needed.

---

## State Space

| Dimension    | Description                              | Bins |
|--------------|------------------------------------------|------|
| t            | Period (month)                           | H    |
| price_bin    | Discretised spot price relative to forward (e.g. 0.7× – 1.3× in 10 steps) | P |
| vol_bin      | Discretised realised volatility (e.g. 5-day RV percentile in 5 buckets)   | V |

Total states: `H × P × V` — for H=12, P=10, V=5 this is 600 states, trivially small.

Price and vol bins are computed from the existing `PriceForecast.paths` array — no
new data needed.

---

## Action Space

At each state the agent chooses a hedge fraction:

```
f ∈ {0, 0.05, 0.10, …, max_hedge}   (step = 0.05, so 21 actions at max_hedge=1.0)
```

This is the same discretisation used by `generate_staggered_candidates()` today.

---

## Bellman Update (backward induction)

```
V*(t=H, pb, vb) = 0   (no cost after horizon)

V*(t, pb, vb) = min over f:
    E[ cost_t(f, spot) + w_cvar_penalty(f, spot) | price_bin=pb, vol_bin=vb ]
    + E[ V*(t+1, pb', vb') | pb, vb, f ]
```

where:
- `cost_t(f, spot) = volume_t × (f × forward_t + (1-f) × spot)`
- `w_cvar_penalty` is the Lagrangian CVaR term above
- `pb', vb'` are the next-period state bins, estimated from transition frequencies
  computed from the simulated paths

Backward induction runs from `t = H` down to `t = 1`. At each step, for each
`(pb, vb)` state, loop over all actions f, compute expected cost + value-to-go,
store the minimising f.

The result is:

```
policy_table[t, pb, vb] → optimal f*
```

---

## Integration with the Existing Pipeline

### What changes

| Component              | Current behaviour                          | After Phase 4                                 |
|------------------------|--------------------------------------------|-----------------------------------------------|
| `node_explore`         | Generates 20+ candidates, simulates, scores | Runs DP → single optimal policy               |
| `generate_batch_candidates` | Returns list of StrategyParams        | Adds `StrategyType.DP_OPTIMAL` candidate      |
| `build_dp_table()`     | Fully implemented, not wired in            | Called from `node_explore`                    |
| Scorer                 | Used to SELECT the winner                  | Used only to REPORT savings vs no-hedge       |
| Arbitrate node         | Picks min(blended_score) across heuristics | Compares DP result vs baseline, writes rationale |
| Risk-Reward chart      | Shows all candidates as bubbles            | DP_OPTIMAL appears as a distinct bubble (expected: lower CVaR, higher mean savings than any heuristic) |

### What stays the same

- All existing strategies (staggered, trigger, vol, hybrid, CVaR-LP) remain in the
  candidate table as **benchmarks** — they show on the risk-reward chart so the demo
  can visually demonstrate that DP dominates them
- `StrategyParams`, `CostResult`, `CandidateRecord` contracts unchanged
- `simulate_cost()` unchanged — used to evaluate the DP policy after the table is built
- Dashboard, HedgeSchedule chart, all frontend unchanged

---

## Implementation Steps

### Step 1 — State binning utility  (`engines/dp_state.py`, new file)

```python
def bin_paths(paths, forward_price, n_price_bins=10, n_vol_bins=5):
    """
    paths: (N, H) simulated spot prices
    Returns:
        price_bins: (N, H) int array  — index into price grid
        vol_bins:   (N, H) int array  — index into vol grid
        transition_matrix: (H, P, V, P, V) float — empirical transition counts
    """
```

Price bins: `spot / forward` ratio clipped to [0.7, 1.3], uniformly bucketed.
Vol bins: rolling 5-day realised volatility percentile (already in `engines/features.py`).

### Step 2 — DP solver  (`engines/dp_optimizer.py`, new file)

```python
def build_cost_dp_table(
    forecast_obj: PriceForecast,
    exposure: ExposureBook,
    risk: RiskAppetite,
    forward_price: float,
    n_price_bins: int = 10,
    n_vol_bins: int = 5,
    action_step: float = 0.05,
) -> np.ndarray:
    """
    Returns policy_table of shape (H, n_price_bins, n_vol_bins)
    with optimal hedge fraction at each state.
    """
```

Internal flow:
1. Call `bin_paths()` to get state sequences and transition matrix
2. Initialise `V = zeros(H+1, P, V)`
3. Backward sweep: for each t from H-1 to 0, for each (pb, vb), evaluate all
   actions and store `argmin` in `policy_table[t, pb, vb]`
4. Return `policy_table`

### Step 3 — Simulate the DP policy  (`engines/cost_simulator.py`, extension)

Add `mode="dp_table"` to `simulate_cost()`:

```python
if params.strategy_type == StrategyType.DP_OPTIMAL:
    # look up f* at each (t, price_bin, vol_bin) for every path
    fractions = lookup_policy(policy_table, price_bins, vol_bins)  # (N, H)
    # then standard vectorised cost computation
```

### Step 4 — Wire into `node_explore`  (`agent/langgraph_workflow.py`)

```python
from engines.dp_optimizer import build_cost_dp_table

dp_table = build_cost_dp_table(
    forecast_obj=forecast_obj,
    exposure=exposure,
    risk=risk,
    forward_price=forward_price,
)

dp_params = StrategyParams(
    strategy_type=StrategyType.DP_OPTIMAL,
    base_fraction=float(np.mean(dp_table)),  # mean fraction for display
    cap=risk.max_hedge,
    dp_table=dp_table,                       # attach table for simulation
)

dp_cost = simulate_cost(
    forecast_obj=forecast_obj,
    exposure=exposure,
    params=dp_params,
    forward_price=forward_price,
    cvar_alpha=risk.cvar_alpha,
)

dp_score = score_policy(cost=dp_cost, no_hedge_cost=no_hedge_cost,
                        params=dp_params, risk=risk)

records.append(CandidateRecord(params=dp_params, score=dp_score, accepted=False))
```

### Step 5 — Contracts update  (`hedging_assistant/contracts.py`)

`StrategyType.DP_OPTIMAL` already exists. Add `dp_table` field to `StrategyParams`:

```python
dp_table: np.ndarray | None = None   # shape (H, P_bins, V_bins), set by DP solver
```

### Step 6 — Remove DP from candidate generator

In `generate_batch_candidates()`, remove the `NotImplementedError` placeholder for
`DP_OPTIMAL` — the DP is now produced directly in `node_explore`, not via the
candidate grid.

---

## Expected Demo Output

On the Risk-Reward chart:
- Staggered, Trigger, Vol, Hybrid, CVaR-LP appear as coloured bubbles
- `DP_OPTIMAL` appears as a **distinct bubble (e.g. cyan)**, positioned lower-left —
  lower CVaR than CVaR-LP, lower E[cost] than any heuristic
- The LLM arbitrate node selects DP_OPTIMAL and the rationale explains why

On the Hedge Schedule chart:
- The DP schedule is non-uniform and state-reactive — higher fractions in high-price
  bins, lower in low-price bins — visually demonstrating the policy adapts to market
  conditions, unlike the flat or smoothly-increasing heuristic schedules

---

## Complexity

| Component           | Complexity                                  |
|---------------------|---------------------------------------------|
| State binning       | O(N × H)                                    |
| Transition matrix   | O(N × H × P × V)                           |
| DP backward sweep   | O(H × P × V × A) where A = actions (~21)   |
| Policy simulation   | O(N × H)                                    |
| **Total**           | **O(N × H × P × V × A)** — for N=1024, H=12, P=10, V=5, A=21: ~13M ops, <1s |

---

## Files Touched

| File                              | Change type |
|-----------------------------------|-------------|
| `engines/dp_state.py`             | New         |
| `engines/dp_optimizer.py`         | New         |
| `engines/cost_simulator.py`       | Extend      |
| `agent/langgraph_workflow.py`     | Extend      |
| `hedging_assistant/contracts.py`  | Extend      |
| `engines/strategy_library.py`     | Remove DP placeholder |
| `api/main.py`                     | No change   |
| Frontend                          | No change   |
