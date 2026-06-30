"""
Data contracts for the agentic hedging decision assistant.

These dataclasses define the input/output shape of every component in the
pipeline.

They are the contracts between stages:
    - Forecaster returns PriceForecast
    - Strategy library consumes StrategyParams and returns HedgingPolicy
    - Cost simulator returns CostDistribution
    - Scorer returns FactorScore
    - Agent returns Recommendation

Phase 3 additions:
    - ForwardCurve
    - RunConfig
    - HYBRID and DP_OPTIMAL strategy types
    - Extended StrategyParams for trigger, volatility, hybrid, and DP policies
    - Recommendation.run_config for auditability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------------

@dataclass
class PriceHistory:
    """
    Raw input: historical crude prices loaded from EIA / Blob / local CSV.
    """

    dates: np.ndarray
    prices: np.ndarray
    symbol: str = "WTI"

    def __post_init__(self):
        self.dates = np.asarray(self.dates)
        self.prices = np.asarray(self.prices, dtype=float)

        if self.prices.ndim != 1:
            raise ValueError("PriceHistory.prices must be a 1D array")

        if len(self.prices) == 0:
            raise ValueError("PriceHistory.prices cannot be empty")

        if len(self.dates) != len(self.prices):
            raise ValueError(
                "PriceHistory.dates and PriceHistory.prices must have same length"
            )

        if np.any(self.prices <= 0):
            raise ValueError("Historical prices must be positive")

        self.symbol = self.symbol.upper()

    def __len__(self) -> int:
        return len(self.prices)


@dataclass
class ExposureBook:
    """
    How much crude the company must buy per future period.
    """

    volumes: np.ndarray
    period_label: str = "month"
    horizon: int = field(init=False)

    def __post_init__(self):
        self.volumes = np.asarray(self.volumes, dtype=float)

        if self.volumes.ndim != 1:
            raise ValueError("ExposureBook.volumes must be a 1D array")

        if len(self.volumes) == 0:
            raise ValueError("ExposureBook.volumes cannot be empty")

        if np.any(self.volumes < 0):
            raise ValueError("Exposure volumes cannot be negative")

        self.horizon = len(self.volumes)


@dataclass
class RiskAppetite:
    """
    Client risk preferences.

    These weights drive the scorer.
    They do not need to sum to 1.
    """

    w_cost: float = 1.0
    w_cvar: float = 1.0
    w_opportunity: float = 0.5
    w_execution: float = 0.25
    cvar_alpha: float = 0.95
    max_hedge: float = 1.0

    def __post_init__(self):
        weights = {
            "w_cost": self.w_cost,
            "w_cvar": self.w_cvar,
            "w_opportunity": self.w_opportunity,
            "w_execution": self.w_execution,
        }

        for name, value in weights.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")

        if not 0 < self.cvar_alpha < 1:
            raise ValueError("cvar_alpha must be between 0 and 1")

        if not 0 <= self.max_hedge <= 1:
            raise ValueError("max_hedge must be between 0 and 1")


@dataclass
class ForwardCurve:
    """
    Forward prices per future period.

    Supports replacing a scalar forward price with a term structure.
    """

    prices: np.ndarray
    frequency: str = "M"
    source: str = "synthetic"
    horizon: int = field(init=False)

    def __post_init__(self):
        self.prices = np.asarray(self.prices, dtype=float)

        if self.prices.ndim != 1:
            raise ValueError("ForwardCurve.prices must be a 1D array")

        if len(self.prices) == 0:
            raise ValueError("ForwardCurve.prices cannot be empty")

        if np.any(self.prices <= 0):
            raise ValueError("Forward prices must be positive")

        self.horizon = len(self.prices)

    @classmethod
    def flat(
        cls,
        price: float,
        horizon: int,
        frequency: str = "M",
        source: str = "synthetic",
    ) -> "ForwardCurve":
        """
        Backward-compatible helper.

        Converts old scalar forward_price behavior into a curve.
        """

        if price <= 0:
            raise ValueError("flat forward price must be positive")

        if horizon <= 0:
            raise ValueError("horizon must be positive")

        return cls(
            prices=np.full(horizon, float(price), dtype=float),
            frequency=frequency,
            source=source,
        )


@dataclass
class RunConfig:
    """
    Reproducibility configuration for one recommendation run.
    """

    n_paths: int = 10_000
    seed: int = 42
    frequency: str = "M"
    calibration_window: Optional[int] = None
    distribution: str = "normal"
    use_regime: bool = False
    cvar_alpha: float = 0.95
    label: str = ""
    model: str = "gbm"

    def __post_init__(self):
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")

        if self.frequency not in {"D", "W", "M"}:
            raise ValueError("frequency must be one of: D, W, M")

        if self.distribution not in {"normal", "student-t"}:
            raise ValueError("distribution must be either 'normal' or 'student-t'")

        if not 0 < self.cvar_alpha < 1:
            raise ValueError("cvar_alpha must be between 0 and 1")

    def to_dict(self) -> dict:
        return {
            "n_paths": self.n_paths,
            "seed": self.seed,
            "frequency": self.frequency,
            "calibration_window": self.calibration_window,
            "distribution": self.distribution,
            "use_regime": self.use_regime,
            "cvar_alpha": self.cvar_alpha,
            "label": self.label,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# FORECASTER
# ---------------------------------------------------------------------------

@dataclass
class PriceForecast:
    """
    Output of the forecaster: simulated future price paths.
    """

    paths: np.ndarray
    model_name: str = "GBM"

    # metadata for explainability / traceability
    start_price: float | None = None
    mu: float | None = None
    sigma: float | None = None
    frequency: str = "M"
    seed: int | None = None
    calibration_window: int | None = None

    horizon: int = field(init=False)
    n_paths: int = field(init=False)

    def __post_init__(self):
        self.paths = np.asarray(self.paths, dtype=float)

        if self.paths.ndim != 2:
            raise ValueError("PriceForecast.paths must be 2D: (n_paths, horizon)")

        if self.paths.shape[0] == 0:
            raise ValueError("PriceForecast.paths cannot have zero paths")

        if self.paths.shape[1] == 0:
            raise ValueError("PriceForecast horizon cannot be zero")

        if np.any(self.paths <= 0):
            raise ValueError("Forecast prices must be positive")

        self.n_paths, self.horizon = self.paths.shape

    def quantiles(self, qs=(10, 50, 90)) -> dict:
        """
        Forecast quantiles per period.
        Feeds dashboard fan chart.
        """

        return {q: np.percentile(self.paths, q, axis=0) for q in qs}


# ---------------------------------------------------------------------------
# STRATEGY LIBRARY
# ---------------------------------------------------------------------------

class StrategyType(str, Enum):
    STAGGERED = "staggered"
    TRIGGER = "trigger"
    VOLATILITY = "volatility"
    HYBRID = "hybrid"
    DP_OPTIMAL = "dp_optimal"
    CVAR_LP = "cvar_lp"


@dataclass
class StrategyParams:
    """
    Parameters that specialize a strategy type into a concrete hedge rule.

    STAGGERED:
        base_fraction, cap

    TRIGGER:
        base_fraction, trigger_threshold, trigger_fraction, ma_window, cap

    VOLATILITY:
        base_fraction, vol_scale_k, ma_window, cap

    HYBRID:
        base_fraction, trigger_threshold, trigger_fraction, vol_scale_k,
        ma_window, cap

    DP_OPTIMAL:
        base_fraction, dp_table, cap

    CVAR_LP:
        fixed_fractions, cap
    """

    strategy_type: StrategyType

    # common
    base_fraction: float = 0.0
    cap: float = 1.0

    # Backward-compatible old trigger fields
    trigger_price: Optional[float] = None
    vol_threshold: Optional[float] = None
    vol_fraction: Optional[float] = None

    # Phase 3 trigger / hybrid
    trigger_threshold: float = 1.0
    trigger_fraction: float = 0.8
    ma_window: int = 5

    # Phase 3 volatility / hybrid
    vol_scale_k: float = 1.0

    # Phase 3 DP optimal
    dp_table: Optional[dict] = None

    # Phase 3 CVaR-LP / fixed schedule
    fixed_fractions: Optional[np.ndarray] = None

    def __post_init__(self):
        if not isinstance(self.strategy_type, StrategyType):
            self.strategy_type = StrategyType(self.strategy_type)

        if not 0 <= self.base_fraction <= 1:
            raise ValueError("base_fraction must be between 0 and 1")

        if not 0 <= self.cap <= 1:
            raise ValueError("cap must be between 0 and 1")

        if not 0 <= self.trigger_fraction <= 1:
            raise ValueError("trigger_fraction must be between 0 and 1")

        if self.trigger_threshold <= 0:
            raise ValueError("trigger_threshold must be positive")

        if self.ma_window < 2:
            raise ValueError("ma_window must be at least 2")

        if self.vol_scale_k < 0:
            raise ValueError("vol_scale_k cannot be negative")

        if self.trigger_price is not None and self.trigger_price <= 0:
            raise ValueError("trigger_price must be positive when provided")

        if self.vol_threshold is not None and self.vol_threshold < 0:
            raise ValueError("vol_threshold cannot be negative")

        if self.vol_fraction is not None and not 0 <= self.vol_fraction <= 1:
            raise ValueError("vol_fraction must be between 0 and 1 when provided")

        if self.fixed_fractions is not None:
            self.fixed_fractions = np.asarray(self.fixed_fractions, dtype=float)

            if self.fixed_fractions.ndim != 1:
                raise ValueError("fixed_fractions must be a 1D array")

            if len(self.fixed_fractions) == 0:
                raise ValueError("fixed_fractions cannot be empty")

            if np.any((self.fixed_fractions < 0) | (self.fixed_fractions > 1)):
                raise ValueError("fixed_fractions must be between 0 and 1")

            if np.any(self.fixed_fractions > self.cap):
                raise ValueError("fixed_fractions cannot exceed cap")
            


@dataclass
class HedgingPolicy:
    """
    Final multi-period hedge policy.
    """

    params: StrategyParams
    hedge_fractions: np.ndarray
    description: str = ""

    def __post_init__(self):
        self.hedge_fractions = np.asarray(self.hedge_fractions, dtype=float)

        if self.hedge_fractions.ndim != 1:
            raise ValueError("HedgingPolicy.hedge_fractions must be 1D")

        if len(self.hedge_fractions) == 0:
            raise ValueError("HedgingPolicy.hedge_fractions cannot be empty")

        if np.any((self.hedge_fractions < 0) | (self.hedge_fractions > 1)):
            raise ValueError("hedge_fractions must be between 0 and 1")


# ---------------------------------------------------------------------------
# MONTE CARLO EVALUATOR
# ---------------------------------------------------------------------------

@dataclass
class CostDistribution:
    """
    Distribution of total procurement cost for one policy.
    """

    costs: np.ndarray
    mean: float = field(init=False)
    p10: float = field(init=False)
    p50: float = field(init=False)
    p90: float = field(init=False)
    cvar: float = field(init=False)
    cvar_alpha: float = 0.95

    ci_mean: tuple[float, float] | None = None
    ci_cvar: tuple[float, float] | None = None

    def __post_init__(self):
        self.costs = np.asarray(self.costs, dtype=float)

        if self.costs.ndim != 1:
            raise ValueError("CostDistribution.costs must be 1D")

        if len(self.costs) == 0:
            raise ValueError("CostDistribution.costs cannot be empty")

        if not 0 < self.cvar_alpha < 1:
            raise ValueError("cvar_alpha must be between 0 and 1")

        self.mean = float(self.costs.mean())
        self.p10 = float(np.percentile(self.costs, 10))
        self.p50 = float(np.percentile(self.costs, 50))
        self.p90 = float(np.percentile(self.costs, 90))

        threshold = np.percentile(self.costs, self.cvar_alpha * 100)
        tail = self.costs[self.costs >= threshold]

        self.cvar = float(tail.mean()) if len(tail) else float(threshold)


# ---------------------------------------------------------------------------
# SCORER
# ---------------------------------------------------------------------------

@dataclass
class FactorScore:
    """
    Four-factor evaluation of one candidate policy.
    """

    cost: float
    cvar: float
    opportunity_cost: float
    execution_risk: float
    blended: float = 0.0

    # Pool-normalized components in [0, 1] (filled by scorer.blend_scores).
    # These make the four factors comparable regardless of dollar magnitude;
    # `blended` becomes the weighted sum of these, i.e. a unitless index.
    cost_norm: float = 0.0
    cvar_norm: float = 0.0
    opportunity_norm: float = 0.0
    execution_norm: float = 0.0

    def __post_init__(self):
        self.cost = float(self.cost)
        self.cvar = float(self.cvar)
        self.opportunity_cost = float(self.opportunity_cost)
        self.execution_risk = float(self.execution_risk)
        self.blended = float(self.blended)
        self.cost_norm = float(self.cost_norm)
        self.cvar_norm = float(self.cvar_norm)
        self.opportunity_norm = float(self.opportunity_norm)
        self.execution_norm = float(self.execution_norm)


# ---------------------------------------------------------------------------
# AGENT OUTPUTS
# ---------------------------------------------------------------------------

@dataclass
class CandidateRecord:
    """
    One strategy considered by the agent.
    """

    params: StrategyParams
    score: FactorScore
    accepted: bool
    note: str = ""


@dataclass
class Recommendation:
    """
    Final recommendation returned by the agent/backend.
    """

    policy: HedgingPolicy
    cost: CostDistribution
    score: FactorScore
    rationale: str
    trace: list[CandidateRecord] = field(default_factory=list)
    assumptions: dict = field(default_factory=dict)
    run_config: Optional[RunConfig] = None


# ---------------------------------------------------------------------------
# BACKTESTER
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Output of walk-forward backtest.
    """

    dates: np.ndarray
    strategy_costs: np.ndarray
    perfect_foresight_costs: np.ndarray
    no_hedge_costs: np.ndarray
    naive_costs: np.ndarray
    captured_fraction: float = 0.0
    label: str = ""

    def __post_init__(self):
        self.dates = np.asarray(self.dates)
        self.strategy_costs = np.asarray(self.strategy_costs, dtype=float)
        self.perfect_foresight_costs = np.asarray(
            self.perfect_foresight_costs,
            dtype=float,
        )
        self.no_hedge_costs = np.asarray(self.no_hedge_costs, dtype=float)
        self.naive_costs = np.asarray(self.naive_costs, dtype=float)

        n = len(self.dates)

        if not (
            len(self.strategy_costs)
            == len(self.perfect_foresight_costs)
            == len(self.no_hedge_costs)
            == len(self.naive_costs)
            == n
        ):
            raise ValueError("All backtest arrays must have the same length")