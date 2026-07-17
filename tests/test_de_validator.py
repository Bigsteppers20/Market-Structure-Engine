"""Tests for decision_engine.validator -- input validation and JSON schema
validation (JSON SCHEMA VALIDATION requirement)."""
from __future__ import annotations

import pytest

from decision_engine.exceptions import MissingAnalysisError
from decision_engine.validator import (
    assert_valid_decision_result_dict,
    validate_decision_inputs,
    validate_decision_result_dict,
)

REQUIRED_TOP_LEVEL = (
    "recommendation", "strategy", "linear_regression", "logistic_regression",
    "trade_plan", "position_size", "market_analysis", "strategy_verdict", "reasoning", "metadata",
)


def _minimal_valid_dict() -> dict:
    return {
        "recommendation": "BUY",
        "strategy": {k: 0.0 for k in (
            "strategy_name", "strategy_recommendation", "market_bias", "strategy_compliance",
            "strategy_confidence", "strategy_validation_score", "forecast_alignment", "probability_alignment",
            "mse_alignment", "consensus_score", "decision_confidence", "opportunity_score", "trade_quality_score",
        )},
        "linear_regression": {k: None for k in (
            "available", "expected_close", "expected_high", "expected_low", "expected_return",
            "expected_pip_movement", "expected_volatility", "expected_MFE", "expected_MAE",
            "prediction_confidence", "prediction_interval",
        )},
        "logistic_regression": {k: None for k in (
            "available", "predicted_class", "buy_probability", "sell_probability", "no_trade_probability",
            "classification_confidence", "probability_margin", "entropy",
        )},
        "trade_plan": {k: None for k in (
            "direction", "entry_price", "stop_loss", "take_profit_1", "take_profit_2", "take_profit_3",
            "risk_reward_ratio", "expected_holding_time", "expected_pip_gain", "expected_maximum_drawdown",
            "target_feasibility", "trade_quality_score",
        )},
        "position_size": {"calculated_by": "RiskManager", "status": "Pending"},
        "market_analysis": {k: None for k in (
            "market_bias", "agreement_level", "forecast_quality", "prediction_stability", "market_regime", "current_trend",
        )},
        "strategy_verdict": {k: None for k in (
            "strategy_name", "overall_strategy_quality", "historical_success_probability",
            "live_market_alignment", "model_alignment", "validation_status", "recommended_action",
        )},
        "reasoning": {k: None for k in (
            "supporting_factors", "opposing_factors", "summary", "why_buy", "why_sell", "why_wait",
            "strategy_compliance_explanation", "forecast_explanation", "probability_explanation",
        )},
        "metadata": {k: None for k in (
            "currency_pair", "timeframe", "timestamp", "strategy_version", "feature_version",
            "market_structure_version", "linear_regression_version", "logistic_regression_version",
            "decision_engine_version",
        )},
    }


def test_validate_decision_inputs_requires_strategy_evaluation() -> None:
    with pytest.raises(MissingAnalysisError):
        validate_decision_inputs(None)


def test_validate_decision_inputs_passes_with_strategy_evaluation() -> None:
    validate_decision_inputs(object())  # must not raise for any non-None value


def test_valid_dict_has_no_issues() -> None:
    assert validate_decision_result_dict(_minimal_valid_dict()) == []


def test_missing_top_level_key_detected() -> None:
    d = _minimal_valid_dict()
    del d["trade_plan"]
    issues = validate_decision_result_dict(d)
    assert any("trade_plan" in i for i in issues)


def test_invalid_recommendation_value_detected() -> None:
    d = _minimal_valid_dict()
    d["recommendation"] = "MAYBE"
    issues = validate_decision_result_dict(d)
    assert any("recommendation" in i for i in issues)


def test_missing_nested_key_detected() -> None:
    d = _minimal_valid_dict()
    del d["strategy"]["consensus_score"]
    issues = validate_decision_result_dict(d)
    assert any("strategy" in i and "consensus_score" in i for i in issues)


@pytest.mark.parametrize("section", REQUIRED_TOP_LEVEL)
def test_every_top_level_section_is_checked(section) -> None:
    d = _minimal_valid_dict()
    del d[section]
    issues = validate_decision_result_dict(d)
    assert issues  # every single required section is actually enforced


def test_assert_valid_raises_on_bad_dict() -> None:
    d = _minimal_valid_dict()
    d["recommendation"] = "BOGUS"
    with pytest.raises(ValueError):
        assert_valid_decision_result_dict(d)


def test_assert_valid_passes_silently_on_good_dict() -> None:
    assert_valid_decision_result_dict(_minimal_valid_dict())  # must not raise
