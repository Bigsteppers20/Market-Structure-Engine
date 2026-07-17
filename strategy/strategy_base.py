"""The strategy interface: every concrete strategy implements ``evaluate()``.

A ``TradingStrategy`` is a named, versioned, weighted collection of rules
(see :mod:`strategy.rule_base`). This module owns the interface itself plus
the deterministic, rule-based (never machine-learned) logic for turning
rule results into a market bias and a trade recommendation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from market_structure import MarketState
from training.utils import utc_timestamp

from .compliance import compute_compliance
from .confidence import compute_confidence
from .config import StrategyConfig
from .explanation import generate_explanations
from .rule_base import Rule, RuleResult, RuleStatus
from .rule_engine import RuleEngine, RuleSpec, count_by_status
from .scoring import compute_scores
from .validator import StrategyValidationError, validate_market_state, validate_strategy_config


class MarketBias(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class TradeRecommendation(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"
    NO_TRADE = "NO_TRADE"


#: Net signed-score thresholds for bucketing into MarketBias -- see
#: compute_market_bias().
_STRONG_THRESHOLD = 0.6
_MILD_THRESHOLD = 0.2

#: A strategy's own rule-vote direction is only trustworthy once the Market
#: Structure Engine's independent trend read confirms it at this level (see
#: compute_mse_compliance()) -- below this, compute_recommendation() forces
#: WAIT no matter how strong the strategy's own compliance/confidence are.
MSE_COMPLIANCE_THRESHOLD = 70.0


def _net_directional_score(rule_results: List[RuleResult]) -> float:
    """Weighted, direction-aware net score in [-1, 1]; 0.0 when nothing
    applicable votes. Shared by compute_market_bias() and
    compute_mse_compliance() so both read the same underlying lean."""
    applicable = [r for r in rule_results if r.status != RuleStatus.NOT_APPLICABLE and r.direction != 0]
    total_weight = sum(r.weight for r in applicable)
    if not applicable or total_weight <= 0:
        return 0.0
    return sum(r.weight * r.direction * (r.score / 100.0) for r in applicable) / total_weight


def compute_market_bias(rule_results: List[RuleResult]) -> MarketBias:
    """Weighted, direction-aware aggregation of every directional rule.

    Each directional rule (``metadata["direction"]`` != 0) contributes
    ``weight * direction * (score / 100)``; the weighted average is bucketed
    into 5 bands. Non-directional rules (support/resistance's raw proximity
    aside, most risk/market-quality rules) simply don't vote here.
    """
    net = _net_directional_score(rule_results)
    if net >= _STRONG_THRESHOLD:
        return MarketBias.STRONG_BULLISH
    if net >= _MILD_THRESHOLD:
        return MarketBias.BULLISH
    if net <= -_STRONG_THRESHOLD:
        return MarketBias.STRONG_BEARISH
    if net <= -_MILD_THRESHOLD:
        return MarketBias.BEARISH
    return MarketBias.NEUTRAL


def compute_mse_compliance(rule_results: List[RuleResult], market_state: MarketState) -> float:
    """0-100: how strongly the Market Structure Engine's own, independently
    computed trend (``market_state.trend`` -- never recomputed here, never
    influenced by the strategy's rules) confirms this strategy's rule-vote
    direction. MSE's read always outranks the strategy's own rules: a
    strategy can only ever claim as much conviction as MSE's trend
    independently backs up.

    - No directional rule-vote, or MSE trend invalid/flat (SIDEWAYS): 50 --
      there is no real MSE-confirmed trend for the strategy to lean on,
      however strong its own rule vote is (this is the Ranging-regime case:
      a strategy can score STRONG_BULLISH on EMA/momentum/MACD alone while
      MSE's actual swing-structure trend is flat).
    - MSE trend direction agrees with the strategy's lean: 50 + strength*50
      (up to 100 at MSE strength 1.0).
    - MSE trend direction disagrees: 50 - strength*50 (down to 0).
    """
    net = _net_directional_score(rule_results)
    strategy_direction = 1 if net > 0 else -1 if net < 0 else 0
    trend = market_state.trend

    if strategy_direction == 0 or trend is None or not trend.valid or int(trend.direction) == 0:
        return 50.0
    if int(trend.direction) == strategy_direction:
        return 50.0 + trend.strength * 50.0
    return 50.0 - trend.strength * 50.0


def compute_recommendation(
    bias: MarketBias, compliance: float, confidence: float, config: StrategyConfig,
    rule_results: List[RuleResult], mse_compliance: float = 100.0,
) -> TradeRecommendation:
    """Purely rule-based decision -- never a model prediction.

    NO_TRADE when there's no directional bias or the (optional) ``"risk"``
    rule failed; BUY/SELL only once compliance, confidence, AND
    ``mse_compliance`` (the Market Structure Engine's own independent trend
    agreement -- see compute_mse_compliance()) all clear their thresholds;
    WAIT for a directionally-biased setup that hasn't (yet) cleared every
    bar. ``mse_compliance`` defaults to 100 (i.e. no gating) for callers that
    don't have a MarketState to check against, such as isolated unit tests.
    """
    risk_failed = any(r.rule_name == "risk" and r.status == RuleStatus.FAIL for r in rule_results)
    if bias == MarketBias.NEUTRAL or risk_failed:
        return TradeRecommendation.NO_TRADE
    if mse_compliance < MSE_COMPLIANCE_THRESHOLD:
        return TradeRecommendation.WAIT
    if compliance >= config.compliance_threshold and confidence >= config.confidence_threshold:
        bullish = bias in (MarketBias.BULLISH, MarketBias.STRONG_BULLISH)
        return TradeRecommendation.BUY if bullish else TradeRecommendation.SELL
    return TradeRecommendation.WAIT


@dataclass(slots=True)
class StrategyEvaluation:
    """Complete output of one ``TradingStrategy.evaluate()`` call."""

    strategy_name: str
    strategy_version: str
    timestamp: str
    symbol: str
    timeframe: str
    market_bias: str
    recommendation: str
    strategy_compliance: float
    strategy_confidence: float
    mse_compliance: float
    technical_score: float
    market_quality_score: float
    risk_quality_score: float
    overall_score: float
    rule_results: List[RuleResult]
    rules_passed: int
    rules_failed: int
    rules_not_applicable: int
    weighted_score: float
    warnings: List[str]
    explanations: List[str]
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name, "strategy_version": self.strategy_version,
            "timestamp": self.timestamp, "symbol": self.symbol, "timeframe": self.timeframe,
            "market_bias": self.market_bias, "recommendation": self.recommendation,
            "strategy_compliance": round(self.strategy_compliance, 2),
            "strategy_confidence": round(self.strategy_confidence, 2),
            "mse_compliance": round(self.mse_compliance, 2),
            "technical_score": round(self.technical_score, 2),
            "market_quality_score": round(self.market_quality_score, 2),
            "risk_quality_score": round(self.risk_quality_score, 2),
            "overall_score": round(self.overall_score, 2),
            "rules_passed": self.rules_passed, "rules_failed": self.rules_failed,
            "rules_not_applicable": self.rules_not_applicable,
            "weighted_score": round(self.weighted_score, 2),
            "warnings": self.warnings,
            "rule_results": [r.to_dict() for r in self.rule_results],
            "explanations": self.explanations,
            "confidence_breakdown": self.confidence_breakdown,
        }


class TradingStrategy(ABC):
    """Abstract strategy: a named, versioned, weighted set of rules.

    Concrete strategies (see the top-level ``strategies/`` package)
    implement :meth:`build_rules`, declaring every rule they know how to
    run and its default constructor parameters. Which of those rules
    actually execute, and with what weight, is entirely controlled by the
    ``StrategyConfig`` passed to ``__init__`` -- the same rule library can
    back many differently-weighted strategy configurations.
    """

    def __init__(self, config: StrategyConfig) -> None:
        validate_strategy_config(config)
        self.config = config
        self._rule_engine = self._build_rule_engine()

    @property
    def name(self) -> str:
        return self.config.strategy_name

    @abstractmethod
    def build_rules(self) -> Dict[str, Rule]:
        """Return ``{rule_name: Rule instance}`` for every rule this strategy
        supports. ``self.config.rule_params`` should be applied here when
        constructing each rule (see concrete strategies for the pattern)."""
        raise NotImplementedError

    def evaluate(self, market_state: MarketState, symbol: str = "UNKNOWN", timeframe: str = "UNKNOWN") -> StrategyEvaluation:
        """Evaluate this strategy against one ``MarketState``. Never inspects
        raw candles, calls the broker, or computes an indicator itself --
        every input comes from ``market_state``."""
        validate_market_state(market_state)

        rule_results = self._rule_engine.run(market_state)
        passed, failed, not_applicable = count_by_status(rule_results)

        compliance = compute_compliance(rule_results)
        confidence_breakdown = compute_confidence(rule_results, market_state)
        confidence = confidence_breakdown.overall
        scores = compute_scores(rule_results, self.config.overall_score_weights)
        bias = compute_market_bias(rule_results)
        mse_compliance = compute_mse_compliance(rule_results, market_state)
        warnings = self._collect_warnings(rule_results, market_state)
        if bias != MarketBias.NEUTRAL and mse_compliance < MSE_COMPLIANCE_THRESHOLD:
            warnings.append(
                f"Market Structure Engine agreement is only {mse_compliance:.0f}% "
                f"(needs {MSE_COMPLIANCE_THRESHOLD:.0f}%) -- {bias.value} bias is the "
                "strategy's own rule vote, not confirmed by MSE's independent trend read, "
                "so BUY/SELL is withheld regardless of compliance/confidence."
            )
        recommendation = compute_recommendation(bias, compliance, confidence, self.config, rule_results, mse_compliance)
        explanations = generate_explanations(rule_results, compliance, confidence, bias.value, warnings, mse_compliance)

        return StrategyEvaluation(
            strategy_name=self.config.strategy_name, strategy_version=self.config.strategy_version,
            timestamp=utc_timestamp(), symbol=symbol, timeframe=timeframe,
            market_bias=bias.value, recommendation=recommendation.value,
            strategy_compliance=compliance, strategy_confidence=confidence,
            mse_compliance=mse_compliance,
            technical_score=scores.technical_score, market_quality_score=scores.market_quality_score,
            risk_quality_score=scores.risk_quality_score, overall_score=scores.overall_score,
            rule_results=rule_results, rules_passed=passed, rules_failed=failed,
            rules_not_applicable=not_applicable, weighted_score=scores.weighted_score,
            warnings=warnings, explanations=explanations,
            confidence_breakdown=confidence_breakdown.to_dict(),
        )

    # ------------------------------------------------------------------ #
    def _build_rule_engine(self) -> RuleEngine:
        available = self.build_rules()
        specs = []
        for rule_name, weight in self.config.rule_weights.items():
            if rule_name not in available:
                raise StrategyValidationError(
                    f"Rule {rule_name!r} in rule_weights is not defined by strategy "
                    f"{self.config.strategy_name!r}. Available rules: {sorted(available)}"
                )
            specs.append(RuleSpec(rule=available[rule_name], weight=weight, enabled=self.config.is_enabled(rule_name)))
        return RuleEngine(specs)

    @staticmethod
    def _collect_warnings(rule_results: List[RuleResult], market_state: MarketState) -> List[str]:
        warnings: List[str] = []
        if market_state.n_candles < 50:
            warnings.append(
                f"Only {market_state.n_candles} candles loaded -- some rules may read as "
                "NOT_APPLICABLE due to indicator warm-up."
            )
        na_count = sum(1 for r in rule_results if r.status == RuleStatus.NOT_APPLICABLE)
        if rule_results and na_count > len(rule_results) / 2:
            na_names = [r.rule_name for r in rule_results if r.status == RuleStatus.NOT_APPLICABLE]
            warnings.append(f"More than half of configured rules are NOT_APPLICABLE: {na_names}")
        risk_result = next((r for r in rule_results if r.rule_name == "risk"), None)
        if risk_result is not None and risk_result.status == RuleStatus.FAIL:
            warnings.append(f"Risk rule failed: {risk_result.reason}")
        return warnings
