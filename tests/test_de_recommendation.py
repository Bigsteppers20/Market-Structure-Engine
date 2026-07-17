"""Tests for decision_engine.recommendation -- consensus voting,
forecast/probability alignment, and the confirm-or-downgrade-never-upgrade
final recommendation rule (STRATEGY ANALYSIS section + the spec's core
"must not override the user's strategy" requirement)."""
from __future__ import annotations

import pytest

from decision_engine.config import DecisionEngineConfig
from decision_engine.recommendation import build_strategy_analysis, compute_final_recommendation


def _analysis(**overrides):
    base = dict(
        strategy_name="trend_following", strategy_recommendation="BUY", market_bias="BULLISH",
        strategy_compliance=80.0, strategy_confidence=75.0, strategy_overall_score=70.0,
        expected_return=0.001, expected_pip_movement=10.0, regression_confidence=80.0,
        predicted_class="BUY", classification_confidence=80.0, config=DecisionEngineConfig(),
    )
    base.update(overrides)
    return build_strategy_analysis(**base)


# --------------------------------------------------------------------------- #
# compute_final_recommendation -- confirm / downgrade / never upgrade
# --------------------------------------------------------------------------- #
def test_no_trade_always_passes_through() -> None:
    result = compute_final_recommendation(
        strategy_recommendation="NO_TRADE", strategy_direction=0, model_net=1.0, config=DecisionEngineConfig(),
    )
    assert result == "NO_TRADE"


def test_wait_never_upgraded_even_with_strong_model_agreement() -> None:
    result = compute_final_recommendation(
        strategy_recommendation="WAIT", strategy_direction=1, model_net=1.0, config=DecisionEngineConfig(),
    )
    assert result == "WAIT"


def test_buy_confirmed_when_models_agree() -> None:
    result = compute_final_recommendation(
        strategy_recommendation="BUY", strategy_direction=1, model_net=0.8, config=DecisionEngineConfig(),
    )
    assert result == "BUY"


def test_buy_confirmed_when_models_have_no_opinion() -> None:
    result = compute_final_recommendation(
        strategy_recommendation="BUY", strategy_direction=1, model_net=0.0, config=DecisionEngineConfig(),
    )
    assert result == "BUY"


def test_buy_downgraded_to_wait_on_strong_model_opposition() -> None:
    result = compute_final_recommendation(
        strategy_recommendation="BUY", strategy_direction=1, model_net=-0.9, config=DecisionEngineConfig(),
    )
    assert result == "WAIT"


def test_sell_downgraded_to_wait_on_strong_model_opposition() -> None:
    result = compute_final_recommendation(
        strategy_recommendation="SELL", strategy_direction=-1, model_net=0.9, config=DecisionEngineConfig(),
    )
    assert result == "WAIT"


def test_buy_not_downgraded_below_opposition_threshold() -> None:
    config = DecisionEngineConfig(downgrade_opposition_threshold=0.9)
    result = compute_final_recommendation(
        strategy_recommendation="BUY", strategy_direction=1, model_net=-0.5, config=config,
    )
    assert result == "BUY"  # opposition present but below the configured threshold


def test_downgrade_never_flips_to_the_opposite_direction() -> None:
    """A downgrade can only ever land on WAIT, never SELL when the strategy said BUY."""
    result = compute_final_recommendation(
        strategy_recommendation="BUY", strategy_direction=1, model_net=-1.0, config=DecisionEngineConfig(),
    )
    assert result in ("BUY", "WAIT")
    assert result != "SELL"


# --------------------------------------------------------------------------- #
# build_strategy_analysis -- consensus / alignment / final recommendation
# --------------------------------------------------------------------------- #
def test_full_agreement_yields_high_consensus_and_confirmed_buy() -> None:
    analysis, final_rec, model_net = _analysis()
    assert final_rec == "BUY"
    assert analysis.consensus_score > 50.0
    assert analysis.forecast_alignment == 100.0
    assert analysis.probability_alignment == 100.0


def test_full_disagreement_downgrades_and_lowers_alignment() -> None:
    analysis, final_rec, model_net = _analysis(
        market_bias="BULLISH", strategy_recommendation="BUY",
        expected_return=-0.001, expected_pip_movement=-10.0, predicted_class="SELL",
    )
    assert analysis.forecast_alignment == 0.0
    assert analysis.probability_alignment == 0.0
    assert final_rec == "WAIT"  # downgraded -- models strongly oppose the strategy


def test_missing_regression_gives_neutral_forecast_alignment() -> None:
    analysis, final_rec, model_net = _analysis(
        expected_return=None, expected_pip_movement=None, regression_confidence=None,
    )
    assert analysis.forecast_alignment == 50.0


def test_missing_classification_gives_neutral_probability_alignment() -> None:
    analysis, final_rec, model_net = _analysis(predicted_class=None, classification_confidence=None)
    assert analysis.probability_alignment == 50.0


def test_strategy_only_still_produces_a_recommendation() -> None:
    analysis, final_rec, model_net = _analysis(
        expected_return=None, expected_pip_movement=None, regression_confidence=None,
        predicted_class=None, classification_confidence=None,
    )
    assert final_rec == "BUY"  # confirmed -- no models available to oppose it
    assert analysis.forecast_alignment == 50.0
    assert analysis.probability_alignment == 50.0


def test_neutral_bias_gives_zero_strategy_direction_vote() -> None:
    analysis, final_rec, model_net = _analysis(
        market_bias="NEUTRAL", strategy_recommendation="NO_TRADE",
    )
    assert final_rec == "NO_TRADE"


def test_decision_confidence_in_valid_range() -> None:
    analysis, _, _ = _analysis()
    assert 0.0 <= analysis.decision_confidence <= 100.0


def test_opportunity_score_in_valid_range() -> None:
    analysis, _, _ = _analysis()
    assert 0.0 <= analysis.opportunity_score <= 100.0


def test_strategy_validation_score_is_mean_of_three_inputs() -> None:
    analysis, _, _ = _analysis(strategy_compliance=90.0, strategy_confidence=60.0, strategy_overall_score=60.0)
    assert analysis.strategy_validation_score == pytest.approx(70.0)


def test_trade_quality_score_starts_at_zero_before_trade_plan_fills_it() -> None:
    analysis, _, _ = _analysis()
    assert analysis.trade_quality_score == 0.0  # filled in by trade_plan_builder afterward
