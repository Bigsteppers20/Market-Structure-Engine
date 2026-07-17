"""Pre/post validation for the Decision Engine.

Two independent checks: are the inputs to :meth:`DecisionEngine.decide`
sufficient (a ``StrategyEvaluation`` is always required), and does a
produced :class:`~decision_engine.decision_result.DecisionResult` serialize
to the expected JSON shape (JSON SCHEMA VALIDATION requirement).
"""
from __future__ import annotations

from typing import Any, Dict, List

from .exceptions import MissingAnalysisError

RECOMMENDATION_VALUES = ("BUY", "SELL", "WAIT", "NO_TRADE")

REQUIRED_TOP_LEVEL_KEYS = (
    "recommendation", "strategy", "linear_regression", "logistic_regression",
    "trade_plan", "position_size", "market_analysis", "strategy_verdict", "reasoning", "metadata",
)

REQUIRED_STRATEGY_KEYS = (
    "strategy_name", "strategy_recommendation", "market_bias", "strategy_compliance",
    "strategy_confidence", "strategy_validation_score", "forecast_alignment", "probability_alignment",
    "mse_alignment", "consensus_score", "decision_confidence", "opportunity_score", "trade_quality_score",
)
REQUIRED_LINEAR_REGRESSION_KEYS = (
    "available", "expected_close", "expected_high", "expected_low", "expected_return",
    "expected_pip_movement", "expected_volatility", "expected_MFE", "expected_MAE",
    "prediction_confidence", "prediction_interval",
)
REQUIRED_LOGISTIC_REGRESSION_KEYS = (
    "available", "predicted_class", "buy_probability", "sell_probability", "no_trade_probability",
    "classification_confidence", "probability_margin", "entropy",
)
REQUIRED_TRADE_PLAN_KEYS = (
    "direction", "entry_price", "stop_loss", "take_profit_1", "take_profit_2", "take_profit_3",
    "risk_reward_ratio", "expected_holding_time", "expected_pip_gain", "expected_maximum_drawdown",
    "target_feasibility", "trade_quality_score",
)
REQUIRED_POSITION_SIZE_KEYS = ("calculated_by", "status")
REQUIRED_MARKET_ANALYSIS_KEYS = (
    "market_bias", "agreement_level", "forecast_quality", "prediction_stability", "market_regime", "current_trend",
)
REQUIRED_STRATEGY_VERDICT_KEYS = (
    "strategy_name", "overall_strategy_quality", "historical_success_probability",
    "live_market_alignment", "model_alignment", "validation_status", "recommended_action",
)
REQUIRED_REASONING_KEYS = (
    "supporting_factors", "opposing_factors", "summary", "why_buy", "why_sell", "why_wait",
    "strategy_compliance_explanation", "forecast_explanation", "probability_explanation",
)
REQUIRED_METADATA_KEYS = (
    "currency_pair", "timeframe", "timestamp", "strategy_version", "feature_version",
    "market_structure_version", "linear_regression_version", "logistic_regression_version",
    "decision_engine_version",
)


def validate_decision_inputs(strategy_evaluation: Any) -> None:
    """The only hard requirement: a ``StrategyEvaluation`` must be supplied
    -- regression/classification predictions are each individually optional."""
    if strategy_evaluation is None:
        raise MissingAnalysisError("DecisionEngine.decide() requires a StrategyEvaluation (strategy_evaluation=None).")


def _missing(d: Dict[str, Any], name: str, keys: tuple) -> List[str]:
    if name not in d or not isinstance(d[name], dict):
        return [f"Missing or non-dict section {name!r}."]
    missing = set(keys) - set(d[name])
    return [f"{name!r} is missing key(s): {sorted(missing)}"] if missing else []


def validate_decision_result_dict(d: Dict[str, Any]) -> List[str]:
    """Structural JSON-schema-style validation of a ``DecisionResult.to_dict()``
    output. Returns a list of issues (empty = valid) -- never raises."""
    issues: List[str] = []
    missing_top = set(REQUIRED_TOP_LEVEL_KEYS) - set(d)
    if missing_top:
        issues.append(f"Missing top-level key(s): {sorted(missing_top)}")
        return issues  # nothing else is safely checkable

    if d["recommendation"] not in RECOMMENDATION_VALUES:
        issues.append(f"recommendation={d['recommendation']!r}, expected one of {RECOMMENDATION_VALUES}.")

    issues += _missing(d, "strategy", REQUIRED_STRATEGY_KEYS)
    issues += _missing(d, "linear_regression", REQUIRED_LINEAR_REGRESSION_KEYS)
    issues += _missing(d, "logistic_regression", REQUIRED_LOGISTIC_REGRESSION_KEYS)
    issues += _missing(d, "trade_plan", REQUIRED_TRADE_PLAN_KEYS)
    issues += _missing(d, "position_size", REQUIRED_POSITION_SIZE_KEYS)
    issues += _missing(d, "market_analysis", REQUIRED_MARKET_ANALYSIS_KEYS)
    issues += _missing(d, "strategy_verdict", REQUIRED_STRATEGY_VERDICT_KEYS)
    issues += _missing(d, "reasoning", REQUIRED_REASONING_KEYS)
    issues += _missing(d, "metadata", REQUIRED_METADATA_KEYS)
    return issues


def assert_valid_decision_result_dict(d: Dict[str, Any]) -> None:
    issues = validate_decision_result_dict(d)
    if issues:
        raise ValueError("Invalid DecisionResult JSON shape:\n  " + "\n  ".join(issues))
