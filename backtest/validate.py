"""
Walk-forward backtester -- VALIDATION HARNESS (offline).

Per the latest scoping decision, the backtester is NOT part of the live
decision flow. It is an offline tool used to VALIDATE AND QUANTIFY UPGRADES:
run it before and after a change (e.g. GBM -> GARCH) to measure whether the
change actually improved out-of-sample performance.

It wraps the whole agent+engine pipeline and replays history with no
look-ahead. SKELETON: signature + contract only; logic filled in Phase 1.
"""

from __future__ import annotations
import numpy as np

from hedging_assistant.contracts import (
    PriceHistory, ExposureBook, RiskAppetite, BacktestResult,
)


def walk_forward_validate(history: PriceHistory,
                          exposure: ExposureBook,
                          risk: RiskAppetite,
                          train_size: int = 1000,
                          test_size: int = 21,
                          step_size: int = 63,
                          label: str = "baseline") -> BacktestResult:
    """
    CONTRACT
      in : full price history, exposure, risk appetite, window sizes, a label
           naming the configuration under test (e.g. "GBM", "GARCH-t")
      out: BacktestResult comparing the system's policy against
           perfect-foresight / no-hedge / naive baselines, plus the headline
           captured_fraction.

    METHOD (Phase 1): walk forward through history; at each decision date give
    the agent ONLY past data (slice at the cutoff -> no look-ahead), let it
    choose a policy, then score that policy on the ACTUAL prices that followed.
    Repeat; aggregate.

    USAGE: run with label="GBM", then re-run with label="GARCH-t", and compare
    captured_fraction to quantify the upgrade.
    """
    raise NotImplementedError(
        "Phase 1: implement walk-forward loop with information cutoff. "
        "Borrow the windowing pattern from skfolio's WalkForward splitter."
    )
