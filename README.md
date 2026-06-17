# Agentic Crude Procurement & Hedging Decision Assistant

Phase 0 skeleton. Pipeline shape and data contracts are in place; baseline
logic runs end-to-end. Advanced logic is filled in later phases.

## Structure
```
hedging_assistant/
  contracts.py            # data contracts (input/output shapes) -- the spine
  data/loader.py          # EIA loader (+ synthetic fallback)
  engines/engines.py      # forecaster, strategy library, Monte Carlo, scorer
  agent/orchestrator.py   # plan-and-execute agent (5 steps)
  backtest/validate.py    # walk-forward VALIDATION harness (offline)
  run_skeleton.py         # end-to-end demo
```

## Run
```
python -m hedging_assistant.run_skeleton          # synthetic data
EIA_API_KEY=yourkey python -m hedging_assistant.run_skeleton   # real WTI
```

## Build status (per phase)
| Component       | Baseline (P1)        | Upgrade (P3)                         |
|-----------------|----------------------|--------------------------------------|
| Forecaster      | GBM (done)           | GARCH-t / Darts / AutoGluon          |
| Strategy lib    | staggered (done)     | + trigger + volatility (stubbed)     |
| Monte Carlo     | plain (done)         | Sobol + antithetic                   |
| Scorer          | 2-factor (done)      | 4-factor + normalisation (stubbed)   |
| Agent           | loop (done)          | LangGraph on Foundry; LLM judgement  |
| Backtester      | contract only        | walk-forward loop (Phase 1)          |

## Get a real EIA key
https://www.eia.gov/opendata/ -- free. Series: WTI=RWTC, Brent=RBRTE.
