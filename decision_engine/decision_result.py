"""``DecisionResult`` -- the single source of truth for the platform.

Every sub-object below is a plain, ``to_dict()``-able dataclass (the same
convention every other engine on this platform uses for its output type --
``RegressionPrediction``, ``ClassificationPrediction``, ``StrategyEvaluation``).
``DecisionResult`` itself never computes anything; it is the assembled
*shape* the rest of ``decision_engine/`` builds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(slots=True)
class StrategyAnalysis:
    """Strategy Engine output plus the Decision Engine's own validation
    scoring (STRATEGY ANALYSIS section)."""

    strategy_name: str
    strategy_recommendation: str
    """The Strategy Engine's own, unmodified recommendation (BUY/SELL/WAIT/
    NO_TRADE) -- never overwritten; see ``DecisionResult.recommendation``
    for the Decision Engine's combined final call."""
    market_bias: str
    strategy_compliance: float
    strategy_confidence: float
    strategy_validation_score: float
    """Composite of compliance/confidence/overall_score -- "how strong is
    this strategy's own read right now", independent of the other two engines."""
    forecast_alignment: float
    """0-100: how well the Linear Regression forecast's direction agrees
    with the strategy's market bias. 50.0 (neutral) if regression wasn't supplied."""
    probability_alignment: float
    """0-100: how well the Logistic Regression predicted class agrees with
    the strategy's recommendation. 50.0 (neutral) if classification wasn't supplied."""
    mse_alignment: float
    """0-100: how strongly the Market Structure Engine's own, independently
    computed trend agrees with the strategy's rule-vote direction -- see
    strategy.strategy_base.compute_mse_compliance(). Passed straight through
    from StrategyEvaluation.mse_compliance; the Strategy Engine already
    enforces the >=70% gate on BUY/SELL before this ever reaches the
    Decision Engine, so this field is for transparency, not re-gating."""
    consensus_score: float
    """0-100: confidence-weighted directional agreement across every
    analytical input actually supplied (1-3 of strategy/regression/classification)."""
    decision_confidence: float
    """0-100: blended confidence in the FINAL recommendation, combining
    each available source's own confidence, weighted by config."""
    opportunity_score: float
    """0-100: how attractive this setup is right now (compliance + overall
    score + consensus + forecast strength)."""
    trade_quality_score: float
    """0-100: same value as ``TradePlan.trade_quality_score`` -- surfaced
    here too since it's part of the spec's STRATEGY ANALYSIS list."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name, "strategy_recommendation": self.strategy_recommendation,
            "market_bias": self.market_bias, "strategy_compliance": round(self.strategy_compliance, 2),
            "strategy_confidence": round(self.strategy_confidence, 2),
            "strategy_validation_score": round(self.strategy_validation_score, 2),
            "forecast_alignment": round(self.forecast_alignment, 2),
            "probability_alignment": round(self.probability_alignment, 2),
            "mse_alignment": round(self.mse_alignment, 2),
            "consensus_score": round(self.consensus_score, 2),
            "decision_confidence": round(self.decision_confidence, 2),
            "opportunity_score": round(self.opportunity_score, 2),
            "trade_quality_score": round(self.trade_quality_score, 2),
        }


@dataclass(slots=True)
class LinearRegressionAnalysis:
    """Linear Regression Engine output, passed through verbatim (LINEAR
    REGRESSION section) -- every field is ``None`` when no
    ``RegressionPrediction`` was supplied to this decision."""

    available: bool
    expected_close: Optional[float] = None
    expected_high: Optional[float] = None
    expected_low: Optional[float] = None
    expected_return: Optional[float] = None
    expected_pip_movement: Optional[float] = None
    expected_volatility: Optional[float] = None
    expected_MFE: Optional[float] = None
    expected_MAE: Optional[float] = None
    prediction_confidence: Optional[float] = None
    prediction_interval: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available, "expected_close": self.expected_close,
            "expected_high": self.expected_high, "expected_low": self.expected_low,
            "expected_return": self.expected_return, "expected_pip_movement": self.expected_pip_movement,
            "expected_volatility": self.expected_volatility, "expected_MFE": self.expected_MFE,
            "expected_MAE": self.expected_MAE,
            "prediction_confidence": None if self.prediction_confidence is None else round(self.prediction_confidence, 2),
            "prediction_interval": {k: list(v) for k, v in self.prediction_interval.items()},
        }


@dataclass(slots=True)
class LogisticRegressionAnalysis:
    """Logistic Regression Engine output, passed through verbatim
    (LOGISTIC REGRESSION section) -- every field is ``None`` when no
    ``ClassificationPrediction`` was supplied to this decision."""

    available: bool
    predicted_class: Optional[str] = None
    buy_probability: Optional[float] = None
    sell_probability: Optional[float] = None
    no_trade_probability: Optional[float] = None
    classification_confidence: Optional[float] = None
    probability_margin: Optional[float] = None
    entropy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available, "predicted_class": self.predicted_class,
            "buy_probability": self.buy_probability, "sell_probability": self.sell_probability,
            "no_trade_probability": self.no_trade_probability,
            "classification_confidence": (
                None if self.classification_confidence is None else round(self.classification_confidence, 2)
            ),
            "probability_margin": self.probability_margin, "entropy": self.entropy,
        }


@dataclass(slots=True)
class TradePlan:
    """Proposed trade structure (TRADE PLAN section). Price levels are
    geometric placements from the already-computed ATR and entry price --
    never a position-size/account-risk calculation (that stays the Risk
    Manager's job, see ``PositionSizePlaceholder``)."""

    direction: str
    """``"BUY"``, ``"SELL"``, or ``"NONE"`` (mirrors ``DecisionResult.recommendation``,
    collapsing WAIT/NO_TRADE to NONE -- there is no trade to plan for either)."""
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    expected_holding_time: str = "N/A"
    expected_pip_gain: Optional[float] = None
    expected_maximum_drawdown: Optional[float] = None
    target_feasibility: float = 0.0
    trade_quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction, "entry_price": self.entry_price, "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1, "take_profit_2": self.take_profit_2,
            "take_profit_3": self.take_profit_3,
            "risk_reward_ratio": None if self.risk_reward_ratio is None else round(self.risk_reward_ratio, 3),
            "expected_holding_time": self.expected_holding_time,
            "expected_pip_gain": None if self.expected_pip_gain is None else round(self.expected_pip_gain, 1),
            "expected_maximum_drawdown": (
                None if self.expected_maximum_drawdown is None else round(self.expected_maximum_drawdown, 1)
            ),
            "target_feasibility": round(self.target_feasibility, 2),
            "trade_quality_score": round(self.trade_quality_score, 2),
        }


@dataclass(slots=True)
class PositionSizePlaceholder:
    """Deliberately NOT computed here -- the Decision Engine must never
    calculate position size or account risk (spec, IMPORTANT section).
    This is a stable, minimal placeholder for the existing Risk Manager to
    fill in."""

    calculated_by: str = "RiskManager"
    status: str = "Pending"

    def to_dict(self) -> Dict[str, str]:
        return {"calculated_by": self.calculated_by, "status": self.status}


@dataclass(slots=True)
class MarketAnalysis:
    """Cross-engine market read (MARKET ANALYSIS section)."""

    market_bias: str
    agreement_level: str
    """Qualitative label derived from ``StrategyAnalysis.consensus_score``
    (e.g. "Strong Agreement"/"Partial Agreement"/"Conflicting"/"No Signal")."""
    forecast_quality: str
    """Qualitative label derived from the regression prediction's own
    confidence and interval width -- "Unavailable" if no regression supplied."""
    prediction_stability: float
    """0-100, averaged from whichever engines' own confidence breakdowns
    expose a ``prediction_stability`` factor (both do). 50.0 (neutral) if neither is available."""
    market_regime: str
    current_trend: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_bias": self.market_bias, "agreement_level": self.agreement_level,
            "forecast_quality": self.forecast_quality,
            "prediction_stability": round(self.prediction_stability, 2),
            "market_regime": self.market_regime, "current_trend": self.current_trend,
        }


@dataclass(slots=True)
class StrategyVerdict:
    """Validates the user's own Strategy Lab strategy against the live
    market and the two ML engines -- never overrides it (STRATEGY VERDICT
    section)."""

    strategy_name: str
    overall_strategy_quality: float
    historical_success_probability: Optional[float]
    """Requires a historical trade/backtest log this platform does not yet
    maintain -- ``None`` unless the caller explicitly supplies
    ``historical_win_rate=`` to ``DecisionEngine.decide()`` (e.g. from a
    future backtest engine or trade journal). Never fabricated."""
    live_market_alignment: float
    model_alignment: float
    validation_status: str
    recommended_action: str
    """Advice about the STRATEGY's fitness (e.g. "review rule weights"), never
    a BUY/SELL/WAIT trade instruction -- that is ``DecisionResult.recommendation``."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "overall_strategy_quality": round(self.overall_strategy_quality, 2),
            "historical_success_probability": self.historical_success_probability,
            "live_market_alignment": round(self.live_market_alignment, 2),
            "model_alignment": round(self.model_alignment, 2),
            "validation_status": self.validation_status,
            "recommended_action": self.recommended_action,
        }


@dataclass(slots=True)
class Explainability:
    """Deterministic, structured reasoning (EXPLAINABILITY section)."""

    supporting_factors: List[str] = field(default_factory=list)
    opposing_factors: List[str] = field(default_factory=list)
    summary: str = ""
    why_buy: str = ""
    why_sell: str = ""
    why_wait: str = ""
    strategy_compliance_explanation: str = ""
    forecast_explanation: str = ""
    probability_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supporting_factors": self.supporting_factors, "opposing_factors": self.opposing_factors,
            "summary": self.summary, "why_buy": self.why_buy, "why_sell": self.why_sell,
            "why_wait": self.why_wait, "strategy_compliance_explanation": self.strategy_compliance_explanation,
            "forecast_explanation": self.forecast_explanation, "probability_explanation": self.probability_explanation,
        }


@dataclass(slots=True)
class DecisionMetadata:
    """Versioning and identification (METADATA section) -- every version
    field is read from the upstream engine's own output, never recomputed."""

    currency_pair: str
    timeframe: str
    timestamp: str
    strategy_version: str
    feature_version: str
    market_structure_version: str
    linear_regression_version: str
    logistic_regression_version: str
    decision_engine_version: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "currency_pair": self.currency_pair, "timeframe": self.timeframe, "timestamp": self.timestamp,
            "strategy_version": self.strategy_version, "feature_version": self.feature_version,
            "market_structure_version": self.market_structure_version,
            "linear_regression_version": self.linear_regression_version,
            "logistic_regression_version": self.logistic_regression_version,
            "decision_engine_version": self.decision_engine_version,
        }


@dataclass(slots=True)
class DecisionResult:
    """The complete, single-source-of-truth output of one
    ``DecisionEngine.decide()`` call.

    Backward compatibility contract: every field here is additive relative
    to a bare ``recommendation``-only response -- a consumer reading only
    ``.recommendation`` (or ``to_dict()["recommendation"]``) continues to
    work unchanged; every other field/section is new and optional to consume.
    """

    recommendation: str
    """The Decision Engine's own combined final call: BUY/SELL/WAIT/NO_TRADE.
    See ``recommendation.py`` for how this is derived -- it can only confirm
    or downgrade the Strategy Engine's own recommendation, never upgrade it."""
    strategy: StrategyAnalysis
    linear_regression: LinearRegressionAnalysis
    logistic_regression: LogisticRegressionAnalysis
    trade_plan: TradePlan
    position_size: PositionSizePlaceholder
    market_analysis: MarketAnalysis
    strategy_verdict: StrategyVerdict
    reasoning: Explainability
    metadata: DecisionMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "strategy": self.strategy.to_dict(),
            "linear_regression": self.linear_regression.to_dict(),
            "logistic_regression": self.logistic_regression.to_dict(),
            "trade_plan": self.trade_plan.to_dict(),
            "position_size": self.position_size.to_dict(),
            "market_analysis": self.market_analysis.to_dict(),
            "strategy_verdict": self.strategy_verdict.to_dict(),
            "reasoning": self.reasoning.to_dict(),
            "metadata": self.metadata.to_dict(),
        }
