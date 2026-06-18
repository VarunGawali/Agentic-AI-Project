"""
Data contracts for the agentic hedging decision assistant.

These dataclasses define the INPUT/OUTPUT shape of every component in the
pipeline. They are the "contracts" between stages: the forecaster promises to
return a PriceForecast, the simulator consumes a HedgingPolicy + PriceForecast
and returns a CostDistribution, and so on.

Designing these first means every component can be built and tested against a
stable interface, and each becomes trivial to expose as an Azure Function later
(every contract is JSON-serialisable via asdict()).

Nothing here contains logic -- only the shapes of the data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


# ----------------------------------------------------------------------------
# INPUTS
# ----------------------------------------------------------------------------

@dataclass
class PriceHistory:
    """Raw input: historical crude prices loaded from EIA."""
    dates: np.ndarray          # shape (T,)   datetime64
    prices: np.ndarray         # shape (T,)   float, USD/barrel
    symbol: str = "WTI"        # "WTI" or "BRENT"

    def __len__(self) -> int:
        return len(self.prices)


@dataclass
class ExposureBook:
    """How much crude the company must buy, per future period."""
    volumes: np.ndarray        # shape (H,)   barrels required each period
    period_label: str = "month"   # the unit of a period (month / week)
    horizon: int = field(init=False)

    def __post_init__(self):
        self.horizon = len(self.volumes)


@dataclass
class RiskAppetite:
    """The client's risk preferences -- drives the scorer's factor weights."""
    # weights for the 4-factor score (need not sum to 1; normalised internally)
    w_cost: float = 1.0
    w_cvar: float = 1.0
    w_opportunity: float = 0.5
    w_execution: float = 0.25
    cvar_alpha: float = 0.95   # tail level for CVaR (worst 5%)
    max_hedge: float = 1.0     # hard limit: never hedge more than this fraction


# ----------------------------------------------------------------------------
# FORECASTER:  PriceHistory -> PriceForecast
# ----------------------------------------------------------------------------

@dataclass
class PriceForecast:
    """
    Output of the forecaster: an ensemble of simulated future price paths.
    This is what the Monte Carlo evaluator samples over.
    """
    paths: np.ndarray          # shape (n_paths, horizon) simulated prices
    horizon: int = field(init=False)
    n_paths: int = field(init=False)
    model_name: str = "GBM"    # which model produced this (GBM, GARCH-t, ...)
    frequency: str = "D"       # "D" daily, "W" weekly, "M" monthly
    calibration_window: Optional[int] = None  # rows of history used to fit drift/vol

    def __post_init__(self):
        self.n_paths, self.horizon = self.paths.shape

    def quantiles(self, qs=(10, 50, 90)) -> dict:
        """P10/P50/P90 per future period -- feeds the fan chart."""
        return {q: np.percentile(self.paths, q, axis=0) for q in qs}


# ----------------------------------------------------------------------------
# STRATEGY LIBRARY:  (params) -> HedgingPolicy applied along a path
# ----------------------------------------------------------------------------

class StrategyType(str, Enum):
    STAGGERED = "staggered"
    TRIGGER = "trigger"
    VOLATILITY = "volatility"


@dataclass
class StrategyParams:
    """Parameters that specialise a strategy type into a concrete rule."""
    strategy_type: StrategyType
    # staggered: uses base_fraction
    # trigger:   uses base_fraction + trigger_price + trigger_fraction
    # volatility:uses base_fraction + vol_threshold + vol_fraction
    base_fraction: float = 0.0
    trigger_price: Optional[float] = None
    trigger_fraction: Optional[float] = None
    vol_threshold: Optional[float] = None
    vol_fraction: Optional[float] = None
    cap: float = 1.0           # max cumulative hedge fraction allowed


@dataclass
class HedgingPolicy:
    """
    The OUTPUT a user ultimately cares about: a full multi-period policy.
    `hedge_fractions` is the realised hedge per period for a representative
    path; `params` is the rule that generates it (so it generalises to any
    path inside the simulator).
    """
    params: StrategyParams
    hedge_fractions: np.ndarray     # shape (horizon,) representative schedule
    description: str = ""           # human-readable, e.g. "40% staggered + ..."


# ----------------------------------------------------------------------------
# MONTE CARLO EVALUATOR:  (PriceForecast, ExposureBook, StrategyParams,
#                          forward_price) -> CostDistribution
# ----------------------------------------------------------------------------

@dataclass
class CostDistribution:
    """Output of the simulator: the distribution of total cost for one policy."""
    costs: np.ndarray          # shape (n_paths,) total cost per path
    mean: float = field(init=False)
    p10: float = field(init=False)
    p50: float = field(init=False)
    p90: float = field(init=False)
    cvar: float = field(init=False)
    cvar_alpha: float = 0.95

    def __post_init__(self):
        self.mean = float(self.costs.mean())
        self.p10 = float(np.percentile(self.costs, 10))
        self.p50 = float(np.percentile(self.costs, 50))
        self.p90 = float(np.percentile(self.costs, 90))
        thr = np.percentile(self.costs, self.cvar_alpha * 100)
        tail = self.costs[self.costs >= thr]
        self.cvar = float(tail.mean()) if len(tail) else float(thr)


# ----------------------------------------------------------------------------
# SCORER:  (CostDistribution + extras) -> FactorScore
# ----------------------------------------------------------------------------

@dataclass
class FactorScore:
    """The 4-factor evaluation of one candidate policy."""
    cost: float                # expected cost (lower better)
    cvar: float                # tail risk (lower better)
    opportunity_cost: float    # regret vs no-hedge when prices fell (lower better)
    execution_risk: float      # operational difficulty proxy (lower better)
    blended: float = 0.0       # weighted, normalised composite (lower better)


# ----------------------------------------------------------------------------
# AGENT:  (everything) -> Recommendation
# ----------------------------------------------------------------------------

@dataclass
class CandidateRecord:
    """One strategy the agent considered -- for the decision trace."""
    params: StrategyParams
    score: FactorScore
    accepted: bool
    note: str = ""


@dataclass
class Recommendation:
    """
    Final output of the agent, handed to governance + the interface.
    Contains the chosen policy, its numbers, and the full decision trace.
    """
    policy: HedgingPolicy
    cost: CostDistribution
    score: FactorScore
    rationale: str                          # plain-English explanation
    trace: list[CandidateRecord] = field(default_factory=list)
    assumptions: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# BACKTESTER (validation harness):  -> BacktestResult
# ----------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Output of the walk-forward backtest -- used OFFLINE to validate and
    quantify upgrades (e.g. GBM vs GARCH), not in the live decision flow.
    """
    dates: np.ndarray                  # decision dates tested
    strategy_costs: np.ndarray         # realised cost of the system's policy
    perfect_foresight_costs: np.ndarray
    no_hedge_costs: np.ndarray
    naive_costs: np.ndarray
    captured_fraction: float = 0.0     # headline: % of perfect-foresight value
    label: str = ""                    # e.g. "GARCH-t v1" -- what was tested
