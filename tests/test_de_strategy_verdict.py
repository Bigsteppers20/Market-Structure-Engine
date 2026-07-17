"""Tests for decision_engine.strategy_verdict -- STRATEGY VERDICT section.

Must never override the user's strategy -- only validate it."""
from __future__ import annotations

from decision_engine.strategy_verdict import build_strategy_verdict


def _verdict(**overrides):
    base = dict(
        strategy_name="trend_following", strategy_overall_score=75.0, strategy_compliance=80.0,
        market_quality_score=70.0, forecast_alignment=100.0, probability_alignment=100.0,
        consensus_score=80.0, historical_win_rate=None,
    )
    base.update(overrides)
    return build_strategy_verdict(**base)


def test_validated_when_compliant_and_consensus_high() -> None:
    verdict = _verdict(strategy_compliance=80.0, consensus_score=80.0)
    assert verdict.validation_status == "Validated"


def test_conflicting_when_compliant_but_models_disagree() -> None:
    verdict = _verdict(strategy_compliance=80.0, consensus_score=10.0)
    assert verdict.validation_status == "Conflicting"


def test_not_validated_when_compliance_very_low() -> None:
    verdict = _verdict(strategy_compliance=20.0, consensus_score=80.0)
    assert verdict.validation_status == "Not Validated"


def test_partially_validated_for_mixed_middling_signals() -> None:
    verdict = _verdict(strategy_compliance=50.0, consensus_score=50.0)
    assert verdict.validation_status == "Partially Validated"


def test_recommended_action_matches_validation_status() -> None:
    verdict = _verdict(strategy_compliance=80.0, consensus_score=80.0)
    assert "continue" in verdict.recommended_action.lower()


def test_historical_success_probability_none_by_default() -> None:
    verdict = _verdict(historical_win_rate=None)
    assert verdict.historical_success_probability is None


def test_historical_success_probability_passed_through_when_supplied() -> None:
    verdict = _verdict(historical_win_rate=0.63)
    assert verdict.historical_success_probability == 0.63


def test_overall_strategy_quality_is_the_strategy_overall_score() -> None:
    verdict = _verdict(strategy_overall_score=88.0)
    assert verdict.overall_strategy_quality == 88.0


def test_live_market_alignment_averages_compliance_and_market_quality() -> None:
    verdict = _verdict(strategy_compliance=80.0, market_quality_score=60.0)
    assert verdict.live_market_alignment == 70.0


def test_model_alignment_averages_forecast_and_probability_alignment() -> None:
    verdict = _verdict(forecast_alignment=100.0, probability_alignment=0.0)
    assert verdict.model_alignment == 50.0


def test_never_overrides_the_users_strategy_name() -> None:
    """The verdict must be attributed to and identify the user's own
    strategy -- it never substitutes a different one."""
    verdict = _verdict(strategy_name="my_custom_strategy")
    assert verdict.strategy_name == "my_custom_strategy"


def test_to_dict_serializable() -> None:
    d = _verdict().to_dict()
    assert set(d) == {
        "strategy_name", "overall_strategy_quality", "historical_success_probability",
        "live_market_alignment", "model_alignment", "validation_status", "recommended_action",
    }
