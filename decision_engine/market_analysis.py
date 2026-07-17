"""Cross-engine market read (MARKET ANALYSIS section).

``market_regime``/``current_trend`` are read directly from already-computed
``MarketState`` fields (``trend``, ``volatility``) -- never a recomputed
indicator or a fresh structure detection.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from market_structure import MarketState

from .decision_result import MarketAnalysis


def _agreement_level(consensus_score: float) -> str:
    if consensus_score >= 75.0:
        return "Strong Agreement"
    if consensus_score >= 40.0:
        return "Partial Agreement"
    if consensus_score > 0.0:
        return "Weak Agreement"
    return "No Clear Signal"


def _forecast_quality(regression_available: bool, prediction_confidence: Optional[float]) -> str:
    if not regression_available or prediction_confidence is None:
        return "Unavailable"
    if prediction_confidence >= 70.0:
        return "High"
    if prediction_confidence >= 40.0:
        return "Moderate"
    return "Low"


def _prediction_stability(
    regression_confidence_breakdown: Optional[Dict[str, Dict[str, float]]],
    classification_confidence_breakdown: Optional[Dict[str, float]],
) -> float:
    values = []
    if regression_confidence_breakdown:
        per_target = [
            v.get("prediction_stability") for v in regression_confidence_breakdown.values()
            if isinstance(v, dict) and v.get("prediction_stability") is not None
        ]
        if per_target:
            values.append(sum(per_target) / len(per_target))
    if classification_confidence_breakdown and classification_confidence_breakdown.get("prediction_stability") is not None:
        values.append(classification_confidence_breakdown["prediction_stability"])
    return sum(values) / len(values) if values else 50.0


def _market_regime(market_state: MarketState) -> str:
    vol = market_state.volatility
    trend = market_state.trend
    if vol is not None and vol.valid and vol.expansion:
        return "High Volatility"
    if vol is not None and vol.valid and vol.compression:
        return "Low Volatility"
    if trend is not None and trend.valid and trend.strength >= 0.5:
        return "Trending"
    return "Ranging"


def _current_trend(market_state: MarketState) -> str:
    trend = market_state.trend
    if trend is None:
        return "UNKNOWN"
    return trend.direction.name


def build_market_analysis(
    *, market_state: MarketState, market_bias: str, consensus_score: float,
    regression_available: bool, regression_prediction_confidence: Optional[float],
    regression_confidence_breakdown: Optional[Dict[str, Dict[str, float]]],
    classification_confidence_breakdown: Optional[Dict[str, float]],
) -> MarketAnalysis:
    return MarketAnalysis(
        market_bias=market_bias, agreement_level=_agreement_level(consensus_score),
        forecast_quality=_forecast_quality(regression_available, regression_prediction_confidence),
        prediction_stability=_prediction_stability(regression_confidence_breakdown, classification_confidence_breakdown),
        market_regime=_market_regime(market_state), current_trend=_current_trend(market_state),
    )
