"""Tests for decision_engine.explainability -- EXPLAINABILITY section.

Every explanation must be deterministic (identical inputs -> identical
strings) and derived only from already-computed values."""
from __future__ import annotations

from decision_engine.decision_result import LinearRegressionAnalysis, LogisticRegressionAnalysis
from decision_engine.explainability import build_explainability


def _lr(**overrides):
    base = dict(available=True, expected_pip_movement=15.0, prediction_confidence=70.0)
    base.update(overrides)
    return LinearRegressionAnalysis(**base)


def _lgr(**overrides):
    base = dict(
        available=True, predicted_class="BUY", buy_probability=0.6, sell_probability=0.2,
        no_trade_probability=0.2, classification_confidence=65.0,
    )
    base.update(overrides)
    return LogisticRegressionAnalysis(**base)


def _build(**overrides):
    base = dict(
        final_recommendation="BUY", strategy_recommendation="BUY", market_bias="BULLISH",
        strategy_name="trend_following", strategy_compliance=80.0, strategy_confidence=75.0,
        linear_regression=_lr(), logistic_regression=_lgr(), forecast_alignment=100.0,
        probability_alignment=100.0, consensus_score=80.0,
    )
    base.update(overrides)
    return build_explainability(**base)


def test_deterministic_given_identical_inputs() -> None:
    result1 = _build()
    result2 = _build()
    assert result1.to_dict() == result2.to_dict()


def test_supporting_factors_include_agreement() -> None:
    result = _build(forecast_alignment=100.0, probability_alignment=100.0, strategy_compliance=80.0)
    assert any("agrees" in f.lower() for f in result.supporting_factors)
    assert any("compliance is high" in f.lower() for f in result.supporting_factors)


def test_opposing_factors_include_disagreement() -> None:
    result = _build(forecast_alignment=0.0, probability_alignment=0.0, strategy_compliance=20.0)
    assert any("opposes" in f.lower() for f in result.opposing_factors)
    assert any("compliance is low" in f.lower() for f in result.opposing_factors)


def test_opposing_factors_note_downgrade() -> None:
    result = _build(final_recommendation="WAIT", strategy_recommendation="BUY")
    assert any("downgraded" in f.lower() for f in result.opposing_factors)


def test_no_downgrade_note_when_confirmed() -> None:
    result = _build(final_recommendation="BUY", strategy_recommendation="BUY")
    assert not any("downgraded" in f.lower() for f in result.opposing_factors)


def test_summary_mentions_final_and_strategy_recommendation() -> None:
    result = _build(final_recommendation="WAIT", strategy_recommendation="BUY", consensus_score=42.0)
    assert "WAIT" in result.summary
    assert "BUY" in result.summary
    assert "42" in result.summary


def test_why_buy_cites_bullish_evidence() -> None:
    result = _build(market_bias="BULLISH", linear_regression=_lr(expected_pip_movement=20.0), logistic_regression=_lgr(buy_probability=0.7))
    assert "strategy bias is BULLISH" in result.why_buy
    assert "+20.0 pips" in result.why_buy or "20.0 pips" in result.why_buy
    assert "70.0%" in result.why_buy


def test_why_sell_cites_bearish_evidence() -> None:
    result = _build(
        market_bias="BEARISH", linear_regression=_lr(expected_pip_movement=-20.0),
        logistic_regression=_lgr(predicted_class="SELL", sell_probability=0.7),
    )
    assert "strategy bias is BEARISH" in result.why_sell
    assert "-20.0 pips" in result.why_sell


def test_why_wait_notes_bias_mismatch_when_neutral() -> None:
    result = _build(market_bias="NEUTRAL", linear_regression=_lr(available=False), logistic_regression=_lgr(available=False))
    assert "NEUTRAL" in result.why_wait


def test_why_wait_cites_mse_alignment_when_below_threshold() -> None:
    result = _build(
        final_recommendation="WAIT", strategy_recommendation="WAIT", market_bias="STRONG_BULLISH",
        mse_alignment=50.0,
    )
    assert "50%" in result.why_wait
    assert "Market Structure Engine" in result.why_wait


def test_why_wait_falls_back_to_generic_bias_message_when_mse_confirms() -> None:
    result = _build(
        final_recommendation="WAIT", strategy_recommendation="WAIT", market_bias="STRONG_BULLISH",
        mse_alignment=90.0,
    )
    assert "Market Structure Engine" not in result.why_wait
    assert "does not clearly support" in result.why_wait


def test_supporting_factors_include_mse_confirmation() -> None:
    result = _build(market_bias="BULLISH", mse_alignment=85.0)
    assert any("Market Structure Engine" in f and "confirms" in f for f in result.supporting_factors)


def test_opposing_factors_include_mse_disagreement() -> None:
    result = _build(market_bias="BULLISH", mse_alignment=40.0)
    assert any("Market Structure Engine" in f and "NOT confirm" in f for f in result.opposing_factors)


def test_why_buy_falls_back_to_no_evidence_message() -> None:
    result = _build(
        market_bias="BEARISH", linear_regression=LinearRegressionAnalysis(available=False),
        logistic_regression=LogisticRegressionAnalysis(available=False),
    )
    assert "No strong evidence" in result.why_buy


def test_forecast_explanation_unavailable_without_regression() -> None:
    result = _build(linear_regression=LinearRegressionAnalysis(available=False))
    assert "not supplied" in result.forecast_explanation


def test_forecast_explanation_reports_direction_and_confidence() -> None:
    result = _build(linear_regression=_lr(expected_pip_movement=-12.5, prediction_confidence=55.0))
    assert "12.5 pip loss" in result.forecast_explanation
    assert "55.0%" in result.forecast_explanation


def test_probability_explanation_unavailable_without_classification() -> None:
    result = _build(logistic_regression=LogisticRegressionAnalysis(available=False))
    assert "not supplied" in result.probability_explanation


def test_probability_explanation_reports_all_three_probabilities() -> None:
    result = _build(logistic_regression=_lgr(buy_probability=0.5, sell_probability=0.3, no_trade_probability=0.2))
    assert "50.0%" in result.probability_explanation
    assert "30.0%" in result.probability_explanation
    assert "20.0%" in result.probability_explanation


def test_strategy_compliance_explanation_includes_name_bias_and_scores() -> None:
    result = _build(strategy_name="my_strat", market_bias="BULLISH", strategy_compliance=80.0, strategy_confidence=75.0)
    assert "my_strat" in result.strategy_compliance_explanation
    assert "BULLISH" in result.strategy_compliance_explanation
    assert "80.0%" in result.strategy_compliance_explanation


def test_to_dict_serializable() -> None:
    d = _build().to_dict()
    assert set(d) == {
        "supporting_factors", "opposing_factors", "summary", "why_buy", "why_sell", "why_wait",
        "strategy_compliance_explanation", "forecast_explanation", "probability_explanation",
    }
