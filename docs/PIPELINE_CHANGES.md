# Pipeline Changes — Forecaster, Scoring, Forward Curve & Control

This document records the changes made to the original hedging pipeline and the
empirical findings that drove them. It is a companion to the code; read it to
understand *why* the pipeline looks the way it does now.

---

## TL;DR of the journey

The original pipeline was: **XGB drift forecast → 6 strategy candidates → 4-factor
scorer picks a winner → LLM arbitrates**. Investigation showed several parts were
either broken or economically vacuous. The pipeline is now:

**ML volatility forecast (unbiased, calibrated) → forward curve (the basis) →
CVaR-LP optimizer (the decision) → LLM as the explain/orchestration layer**, with
the old strategies demoted to **benchmarks** and DP repositioned as **receding-horizon
control (MPC)**.

---

## 1. The two empirical findings that reframed everything

Measured by walk-forward backtest on WTI daily history (2010–2026):

| Signal | Result | Implication |
|---|---|---|
| **Price direction** | Information coefficient ≈ **0** (even in-sample) | Cannot forecast direction → cannot promise price-prediction savings |
| **Volatility** | Trailing→forward vol Spearman IC ≈ **0.4–0.5** | Vol *is* forecastable → point ML there |

Consequence: hedging's expected saving is `f·(E[spot] − F)`. With a driftless
(honest) forecast and `forward = spot`, that is **zero**, so no strategy can beat
no-hedge on expected cost. The only honest sources of value are **(a) the
forward-curve basis** and **(b) risk (CVaR) reduction**.

---

## 2. Forecaster: XGB-GARCH-t (drift) → XGB-Vol-t (volatility)

**Problem found.** The old `xgb-garch-t` pointed XGBoost at *drift* (direction).
It was biased (median terminal drift +12%, mean +200–316% from daily-drift
compounding over 126 steps) and produced **vacuous intervals** (100% of realized
prices fell inside p10–p90 — an interval that is never wrong carries no
information, so its CVaR is meaningless).

**Fix.** New `engines/ml_vol_forecaster.py` (`forecast_ml_vol_t`, "XGB-Vol-t"):
- XGBoost predicts **forward realized volatility** from HAR features (trailing RV
  over 5/21/63/126 days, etc.).
- Prices simulate **driftless** (martingale, `E[spot] ≈ forward`) with Student-t
  shocks scaled by the ML vol × a calibration factor (`vol_scale ≈ 1.25`).

**Result (walk-forward, 115 windows):** median drift **+0.0%** (was +12%), mean
+7.0% matching realized +6.3%, coverage **86%/94%** (was vacuous 100%/100%),
out-of-sample vol IC **+0.27**. Beats naive GBM by **~24% CRPS** (9.85 vs 13.02)
and removes GBM's +2.5% drift bias.

`engines/forecaster.py` now routes `xgb-garch-t` / `xgb-vol-t` / `ml-vol-t` to the
new forecaster; the legacy drift model + id are kept as an alias for compatibility.

---

## 3. Ensemble forecaster (prototype, not yet productionized)

Built a multi-model ensemble matching the "Level anchor + Vol/Tail engine +
Scenario generator + CRPS selector" design: members = {XGB-HAR vol, GJR-GARCH-t
vol (reusing `fit_garch_t_on_residuals`), EWMA/RiskMetrics vol}, mixture combined
by trailing-CRPS dynamic weights, driftless Student-t simulation.

**Finding:** the ensemble **matches** the single XGB-Vol-t on CRPS (the selector
correctly concentrates on the best member) with slightly better tail coverage
(93% vs 91%). Its value is **robustness/insurance across regimes**, not raw
accuracy on this data.

**Productionized** as `engines/ensemble_forecaster.py` (`forecast_ensemble_t`,
"XGB-Ensemble-t"), selectable via `model="ensemble-t"` through the dispatcher /
API / frontend ("XGB Ensemble"). It pools driftless Student-t paths across the
three members (weighted mixture), so **model disagreement widens the tails** — and
because the agent's LP consumes exactly those paths, selecting the ensemble model
*is* the "ensemble distribution → CVaR-LP" robustness wiring.

---

## 4. Scorer: magnitude normalization + opportunity cost dropped

- **`blend_scores()`** (new in `engines/scorer.py`): min-max normalizes each of the
  four factors to [0,1] across the candidate pool before weighting, so cost/CVaR
  (~$50–60M) no longer swamp opportunity/execution (~$0.5–2M). Blended score is now
  a unitless index; `FactorScore` gained `*_norm` fields. Displays updated (API,
  arbitrate table) to show the index, not fake dollars.
- **Opportunity cost** was found to actively harm: `E[max(cost − no_hedge, 0)]`
  rewards staying near no-hedge, which flattens schedules. Dropped from the
  optimized objective (report-only); default `w_opportunity → 0`.

## 4b. Earlier correctness fixes (same session, pre-forecaster work)

- Path-dependent strategies now receive `price_history` + `long_run_vol` through
  `simulate_cost` → `apply_strategy` (they were silently flat before).
- LLM `w_cvar_boost` capped at 50% of the user's weight (was uncapped).
- Arbitrate prompt defaults to the min-score candidate unless there's a strong
  regime reason (was free to override).

---

## 5. Forward curve — the lever that makes hedging save money

**New `engines/forward_curve.py`:**
- `build_parametric_curve(spot, horizon, annual_carry, frequency)` — scenario curve
  `F_t = spot·(1+carry)^(t/ppy)`; `carry<0` backwardation, `>0` contango.
- `build_curve_from_futures(front_contracts, …)` — real EIA WTI futures
  (`RCLC1–4`) front + log-slope tail extrapolation. Loader implemented:
  `data/loader.py` gains `fetch_eia_futures_curve()` (RCLC1–4 from the EIA `fut`
  endpoint, same key as spot) and `load_forward_curve()` (builds the curve, flat
  fallback if the fetch fails). API `use_eia_futures=true` switches from the
  parametric carry to the real curve.
- `implied_annual_carry(...)` for reporting.

**Wired into the live product:**
- API `RecommendRequest` gained `forward_carry`; the handler builds the curve from
  `forward_price` (front anchor) + `forward_carry` and passes the **array** to
  `run_agent` (engines already accept curves via `resolve_forward_curve`). Response
  echoes `forward_curve` + `forward_carry`.
- Frontend: a **"Curve Shape"** slider (backwardation ↔ contango).

**Backtest proof (real WTI paths, driftless XGB-vol forecast, CVaR-LP):**

| Forward curve | avg hedge | savings vs no-hedge |
|---|---|---|
| backwardation −6% | 100% | **+3.31%** |
| flat 0% | 100% | +1.54% (this window's realized drift) |
| contango +6% | 96% (pulls back) | +0.06% (avoids the loss) |

The basis creates a **monotonic, controllable savings gradient**; with cost-priority
weights the optimizer correctly hedges backwardation and abstains in contango.
**This is the mechanism that turns "no strategy beats no-hedge" into real savings.**

---

## 6. DP repositioned → receding-horizon control (MPC)

Standalone DP over price/vol state **degenerates** (no directional skill to exploit;
under a flat curve the optimum is trivially "hedge 100%"), so DP does **not** belong
in the core hedge decision — the convex CVaR-LP owns that.

Its honest home is **`engines/mpc.py` (`receding_horizon_execute`)**: at each period,
re-forecast vol, re-anchor the curve, re-solve the CVaR-LP for the remaining horizon,
execute the front action, advance. That is Bellman-by-re-optimization ("MPC =
approximate DP"), reusing the existing LP.

**Backtest finding (open-loop vs MPC) — they have opposite, economically-coherent
strengths:**

| Forward curve | Open-loop LP savings | MPC savings |
|---|---|---|
| backwardation −6% | **+1.74%** | +0.16% |
| contango +6% | −1.76% | **−0.85%** |

Open-loop locks the *whole* curve at t=0, so it captures the cheap far-month
forwards in **backwardation**. MPC hedges progressively at rolling near-forwards and
stays flexible, so it avoids over-committing to expensive forwards in **contango**.
Practical rule: lock the curve (open-loop) when it's backwardated; prefer MPC's
flexibility when it's in contango.

(A physical **storage/inventory DP** is the one place a true DP beats the LP and cuts
cost — deferred to roadmap; needs storage cost/capacity data.)

---

## 7. Strategic repositioning (no code, but load-bearing)

- **CVaR-LP is the decision.** It directly minimizes cost+CVaR over the forward
  curve; it subsumes the useful behavior of the heuristics.
- **STAGGERED / TRIGGER / VOLATILITY / HYBRID / DP_OPTIMAL → benchmarks.** They
  populate the risk-reward chart to *show the optimizer beats the rules of thumb*
  (trust/explainability). TRIGGER/HYBRID/DP rely on a directional/state edge the
  data says isn't there — keep as chart benchmarks or retire; STAGGERED also serves
  as a simple explainable option.
- **The LLM/agent is repositioned** from co-decision-maker (assess/arbitrate — now
  obsolete, and an LLM should never make the numeric hedge call) to the **interface /
  orchestration layer**: context→optimization-parameters, executive explanation,
  conversational what-if (tool-calling the engine), risk-preference elicitation.

---

## 8. Files added / changed

**Added:** `engines/ml_vol_forecaster.py`, `engines/forward_curve.py`,
`engines/mpc.py`, this doc.
**Changed:** `engines/forecaster.py` (routing), `engines/scorer.py` (`blend_scores`,
opp-cost), `engines/cost_simulator.py` (history threading), `contracts.py`
(`FactorScore.*_norm`), `agent/langgraph_workflow.py` (normalization, arbitrate,
DP candidate), `api/main.py` (forward curve wiring), `frontend/src/App.jsx`
(curve slider), `frontend/src/components/RiskRewardChart.jsx` (DP bubble).

## 9. Roadmap

1. ~~Wire the **ensemble distribution → CVaR-LP**~~ — done (select `ensemble-t`).
2. ~~**EIA `RCLC1–4` futures** loader~~ — done (`load_forward_curve`,
   `use_eia_futures`); needs a live EIA key to exercise the real fetch.
3. **Storage/inventory DP** module (if users have physical storage).
4. Retire the obsolete heuristic strategies / dead legacy modules
   (`engines/engines.py`, `agent/langgraph_agent.py`, `agent/orchestrator.py`).
5. Backtest ensemble-LP vs single-LP realized cost/CVaR (expected: marginal, since
   the ensemble's edge is tail robustness, not point accuracy).
