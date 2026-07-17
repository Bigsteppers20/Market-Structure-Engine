"""Consensus voting and the final combined recommendation.

Deterministic, weighted-vote combination logic -- **not** a new predictive
model. The Decision Engine remains an orchestration layer: every input here
is an already-computed score/direction/confidence read straight from the
Strategy Engine's ``StrategyEvaluation`` and the two ML engines'
``RegressionPrediction``/``ClassificationPrediction``.

Core rule (spec-mandated): the Decision Engine may only **confirm** or
**downgrade** the Strategy Engine's own recommendation -- it never upgrades
a WAIT/NO_TRADE into a trade the strategy itself didn't clear (that would
override the user's strategy, which the STRATEGY VERDICT section
explicitly forbids, and the same principle applies to the final
recommendation itself).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import DecisionEngineConfig
from .decision_result import StrategyAnalysis

_BULLISH_BIASES = ("BULLISH", "STRONG_BULLISH")
_BEARISH_BIASES = ("BEARISH", "STRONG_BEARISH")


@dataclass(slots=True)
class _Vote:
    direction: int
    """-1, 0, or +1."""
    weight: float
    """0-1, this source's own confidence."""
    available: bool


def _strategy_vote(market_bias: str, strategy_confidence: float) -> _Vote:
    if market_bias in _BULLISH_BIASES:
        direction = 1
    elif market_bias in _BEARISH_BIASES:
        direction = -1
    else:
        direction = 0
    return _Vote(direction=direction, weight=max(0.0, min(1.0, strategy_confidence / 100.0)), available=True)


def _regression_vote(expected_return: Optional[float], expected_pip_movement: Optional[float], prediction_confidence: Optional[float]) -> _Vote:
    if prediction_confidence is None:
        return _Vote(direction=0, weight=0.0, available=False)
    value = expected_return if expected_return is not None else expected_pip_movement
    direction = 0 if value is None else (1 if value > 0 else (-1 if value < 0 else 0))
    return _Vote(direction=direction, weight=max(0.0, min(1.0, prediction_confidence / 100.0)), available=True)


def _classification_vote(predicted_class: Optional[str], classification_confidence: Optional[float]) -> _Vote:
    if predicted_class is None or classification_confidence is None:
        return _Vote(direction=0, weight=0.0, available=False)
    direction = 1 if predicted_class == "BUY" else (-1 if predicted_class == "SELL" else 0)
    return _Vote(direction=direction, weight=max(0.0, min(1.0, classification_confidence / 100.0)), available=True)


def _weighted_net(votes: List[_Vote]) -> float:
    present = [v for v in votes if v.available]
    total_weight = sum(v.weight for v in present)
    if not present or total_weight <= 0:
        return 0.0
    return sum(v.direction * v.weight for v in present) / total_weight


def _alignment_score(a: int, b: int) -> float:
    """100 = same nonzero direction, 0 = opposite nonzero directions, 50 =
    either side is neutral (genuinely ambiguous, not a disagreement)."""
    product = a * b
    if product > 0:
        return 100.0
    if product < 0:
        return 0.0
    return 50.0


def _renormalized_blend(values: Dict[str, Optional[float]], weights: Dict[str, float]) -> float:
    present = {k: v for k, v in values.items() if v is not None}
    total_weight = sum(weights.get(k, 0.0) for k in present)
    if not present or total_weight <= 0:
        return 0.0
    return sum(v * weights.get(k, 0.0) for k, v in present.items()) / total_weight


def compute_final_recommendation(
    *, strategy_recommendation: str, strategy_direction: int, model_net: float, config: DecisionEngineConfig,
) -> str:
    """Confirm or downgrade the strategy's own recommendation. Never upgrades."""
    if strategy_recommendation == "NO_TRADE":
        return "NO_TRADE"
    if strategy_recommendation == "WAIT":
        return "WAIT"
    # strategy_recommendation is BUY or SELL here.
    opposes = (model_net * strategy_direction) < 0 and abs(model_net) >= config.downgrade_opposition_threshold
    return "WAIT" if opposes else strategy_recommendation


def build_strategy_analysis(
    *, strategy_name: str, strategy_recommendation: str, market_bias: str, strategy_compliance: float,
    strategy_confidence: float, strategy_overall_score: float,
    expected_return: Optional[float], expected_pip_movement: Optional[float], regression_confidence: Optional[float],
    predicted_class: Optional[str], classification_confidence: Optional[float],
    config: DecisionEngineConfig, mse_compliance: float = 100.0,
) -> Tuple[StrategyAnalysis, str, float]:
    """Build the full :class:`StrategyAnalysis` plus the final combined
    recommendation and the raw model-only net vote (returned for the
    trade-plan builder's direction, and for testing)."""
    strategy_vote = _strategy_vote(market_bias, strategy_confidence)
    regression_vote = _regression_vote(expected_return, expected_pip_movement, regression_confidence)
    classification_vote = _classification_vote(predicted_class, classification_confidence)

    overall_net = _weighted_net([strategy_vote, regression_vote, classification_vote])
    model_net = _weighted_net([regression_vote, classification_vote])
    consensus_score = abs(overall_net) * 100.0

    forecast_alignment = 50.0 if not regression_vote.available else _alignment_score(strategy_vote.direction, regression_vote.direction)
    probability_alignment = 50.0 if not classification_vote.available else _alignment_score(strategy_vote.direction, classification_vote.direction)

    final_recommendation = compute_final_recommendation(
        strategy_recommendation=strategy_recommendation, strategy_direction=strategy_vote.direction,
        model_net=model_net, config=config,
    )

    decision_confidence = _renormalized_blend(
        {"strategy": strategy_confidence, "regression": regression_confidence, "classification": classification_confidence},
        config.decision_confidence_weights,
    )

    strategy_validation_score = (strategy_compliance + strategy_confidence + strategy_overall_score) / 3.0

    opportunity_score = _renormalized_blend(
        {
            "compliance": strategy_compliance, "overall_score": strategy_overall_score,
            "consensus": consensus_score, "forecast_strength": forecast_alignment,
        },
        config.opportunity_score_weights,
    )

    analysis = StrategyAnalysis(
        strategy_name=strategy_name, strategy_recommendation=strategy_recommendation, market_bias=market_bias,
        strategy_compliance=strategy_compliance, strategy_confidence=strategy_confidence,
        strategy_validation_score=strategy_validation_score, forecast_alignment=forecast_alignment,
        probability_alignment=probability_alignment, mse_alignment=mse_compliance, consensus_score=consensus_score,
        decision_confidence=decision_confidence, opportunity_score=opportunity_score,
        trade_quality_score=0.0,  # filled in once trade_plan_builder computes it
    )
    return analysis, final_recommendation, model_net
