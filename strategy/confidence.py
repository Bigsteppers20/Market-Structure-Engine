"""Confidence: how *certain* the engine is about its own reading -- computed
independently from compliance, from six distinct factors.

Compliance answers "how well does the market match the strategy's rules";
confidence answers "how much should you trust that answer right now."  A
strategy can be highly compliant but low-confidence (e.g. every rule agrees,
but the market is in a volatile, unstable regime where that agreement is
fragile) -- these two numbers are computed from different inputs on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from market_structure import MarketState

from .rule_base import RuleResult, RuleStatus


@dataclass(slots=True)
class ConfidenceBreakdown:
    """The six independent components blended into ``strategy_confidence``."""

    rule_certainty: float
    indicator_agreement: float
    trend_consistency: float
    volatility_stability: float
    market_structure_quality: float
    signal_stability: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "rule_certainty": round(self.rule_certainty, 2),
            "indicator_agreement": round(self.indicator_agreement, 2),
            "trend_consistency": round(self.trend_consistency, 2),
            "volatility_stability": round(self.volatility_stability, 2),
            "market_structure_quality": round(self.market_structure_quality, 2),
            "signal_stability": round(self.signal_stability, 2),
        }

    @property
    def overall(self) -> float:
        weights = {
            "rule_certainty": 0.25, "indicator_agreement": 0.20, "trend_consistency": 0.15,
            "volatility_stability": 0.15, "market_structure_quality": 0.15, "signal_stability": 0.10,
        }
        total = sum(getattr(self, k) * w for k, w in weights.items())
        return max(0.0, min(100.0, total))


def _rule_certainty(applicable: List[RuleResult]) -> float:
    if not applicable:
        return 0.0
    return sum(r.confidence for r in applicable) / len(applicable)


def _indicator_agreement(applicable: List[RuleResult]) -> float:
    directions = [r.direction for r in applicable if r.direction != 0]
    if not directions:
        return 50.0  # no directional signal at all -> neutral, not "confident"
    majority_sign = 1 if sum(directions) >= 0 else -1
    agreement_fraction = sum(1 for d in directions if d == majority_sign) / len(directions)
    return agreement_fraction * 100.0


def _trend_consistency(market_state: MarketState) -> float:
    trend = market_state.trend
    if trend is None or not trend.valid:
        return 50.0
    return min(100.0, trend.strength * 100.0)


def _volatility_stability(market_state: MarketState) -> float:
    vol = market_state.volatility
    if vol is None or not vol.valid:
        return 50.0
    if vol.expansion:
        return 40.0
    if vol.compression:
        return 70.0
    return 90.0


def _market_structure_quality(market_state: MarketState) -> float:
    """Rewards having *confirmed*, non-stale structure to lean on."""
    s = market_state.structure
    if s is None or s.last_bos_direction == 0.0:
        return 40.0
    freshness = max(0.0, 1.0 - min(s.bars_since_bos, 50) / 50.0)
    strength_component = min(1.0, s.last_bos_strength / 2.0)
    return (0.5 * freshness + 0.5 * strength_component) * 100.0


def _signal_stability(market_state: MarketState) -> float:
    """Rewards agreement between the most recent BOS and CHOCH direction
    and the prevailing trend direction -- disagreement signals a market
    that's still resolving its own character, i.e. an unstable read."""
    trend = market_state.trend
    s = market_state.structure
    if trend is None or not trend.valid or s is None or s.last_bos_direction == 0.0:
        return 50.0
    agree = int(trend.direction) == int(s.last_bos_direction)
    return 85.0 if agree else 35.0


def compute_confidence(rule_results: List[RuleResult], market_state: MarketState) -> ConfidenceBreakdown:
    """Compute the six-factor confidence breakdown for one evaluation."""
    applicable = [r for r in rule_results if r.status != RuleStatus.NOT_APPLICABLE]
    return ConfidenceBreakdown(
        rule_certainty=_rule_certainty(applicable),
        indicator_agreement=_indicator_agreement(applicable),
        trend_consistency=_trend_consistency(market_state),
        volatility_stability=_volatility_stability(market_state),
        market_structure_quality=_market_structure_quality(market_state),
        signal_stability=_signal_stability(market_state),
    )
