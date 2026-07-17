"""End-to-end tests for decision_engine.DecisionEngine -- combines a real
StrategyEvaluation with real RegressionPrediction/ClassificationPrediction
into one DecisionResult. Covers: serialization, backward compatibility,
trade plan generation, strategy verdict, explainability, metadata, JSON
schema validation, Risk Manager placeholder compatibility, and Agentic AI
(plain-JSON) compatibility."""
from __future__ import annotations

import json

import pytest

from conftest import make_ohlcv
from market_structure import EngineConfig, MarketStructureEngine
from strategy import StrategyEngine
from strategies.trend_following import TrendFollowingStrategy, default_config as trend_default_config

from decision_engine import DecisionEngine, DecisionEngineConfig, DecisionResult
from decision_engine.exceptions import MissingAnalysisError
from decision_engine.validator import validate_decision_result_dict


@pytest.fixture(scope="module")
def de_market_state():
    df = make_ohlcv(700, seed=17)
    engine = MarketStructureEngine(EngineConfig(swing_window=5))
    engine.load(df.iloc[-150:])
    engine.analyze()
    return engine.market_state()


@pytest.fixture(scope="module")
def de_strategy_evaluation(de_market_state):
    strategy_engine = StrategyEngine()
    strategy = TrendFollowingStrategy(trend_default_config())
    strategy_engine.register_strategy(strategy)
    return strategy_engine.evaluate(de_market_state, "trend_following", symbol="EUR_USD", timeframe="M5")


@pytest.fixture(scope="module")
def de_regression_prediction(mon_regression_engine, de_market_state):
    engine, dataset, split, ml_cfg, records = mon_regression_engine
    return engine.predict(de_market_state, symbol="EUR_USD", timeframe="M5")


@pytest.fixture(scope="module")
def de_classification_prediction(mon_classification_engine, de_market_state):
    engine, dataset, split, ml_cfg, record = mon_classification_engine
    return engine.predict(de_market_state, symbol="EUR_USD", timeframe="M5")


# --------------------------------------------------------------------------- #
# Full combination -- real Strategy + real Linear Regression + real Logistic Regression
# --------------------------------------------------------------------------- #
def test_full_decision_with_all_three_engines(
    de_market_state, de_strategy_evaluation, de_regression_prediction, de_classification_prediction,
) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(
        market_state=de_market_state, strategy_evaluation=de_strategy_evaluation,
        regression_prediction=de_regression_prediction, classification_prediction=de_classification_prediction,
        symbol="EUR_USD", timeframe="M5",
    )
    assert isinstance(result, DecisionResult)
    assert result.recommendation in ("BUY", "SELL", "WAIT", "NO_TRADE")
    assert result.linear_regression.available is True
    assert result.logistic_regression.available is True

    d = result.to_dict()
    assert validate_decision_result_dict(d) == []
    # JSON-serializable (JSON SCHEMA VALIDATION / Agentic AI compatibility)
    json.dumps(d)


def test_strategy_only_still_produces_full_shape(de_market_state, de_strategy_evaluation) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5")
    assert result.linear_regression.available is False
    assert result.logistic_regression.available is False
    assert result.linear_regression.expected_close is None
    assert result.logistic_regression.predicted_class is None
    d = result.to_dict()
    assert validate_decision_result_dict(d) == []


def test_missing_strategy_evaluation_raises() -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    with pytest.raises(MissingAnalysisError):
        engine.decide(market_state=object(), strategy_evaluation=None)


# --------------------------------------------------------------------------- #
# Backward compatibility -- a consumer reading only .recommendation works
# --------------------------------------------------------------------------- #
def test_backward_compatible_recommendation_only_consumer(de_market_state, de_strategy_evaluation) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5")

    def legacy_consumer(decision_result) -> str:
        return decision_result.recommendation  # only field a "legacy" consumer reads

    assert legacy_consumer(result) in ("BUY", "SELL", "WAIT", "NO_TRADE")
    assert legacy_consumer(result) == result.to_dict()["recommendation"]


def test_all_original_json_structure_keys_present(de_market_state, de_strategy_evaluation) -> None:
    """Matches the spec's JSON_STRUCTURE example exactly, plus the
    additive position_size section."""
    engine = DecisionEngine(DecisionEngineConfig())
    d = engine.decide(market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5").to_dict()
    for key in (
        "recommendation", "strategy", "linear_regression", "logistic_regression",
        "trade_plan", "market_analysis", "strategy_verdict", "reasoning", "metadata",
    ):
        assert key in d


# --------------------------------------------------------------------------- #
# Trade Plan / Position Size (Risk Manager compatibility)
# --------------------------------------------------------------------------- #
def test_trade_plan_present_for_active_recommendation(de_market_state, de_strategy_evaluation) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5")
    if result.recommendation in ("BUY", "SELL"):
        assert result.trade_plan.entry_price is not None
        assert result.trade_plan.stop_loss is not None
        assert result.trade_plan.take_profit_1 is not None
        assert result.trade_plan.take_profit_2 is not None
        assert result.trade_plan.take_profit_3 is not None
    else:
        assert result.trade_plan.direction == "NONE"


def test_position_size_is_risk_manager_placeholder(de_market_state, de_strategy_evaluation) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5")
    assert result.position_size.to_dict() == {"calculated_by": "RiskManager", "status": "Pending"}


def test_position_size_never_computed_regardless_of_recommendation(
    de_market_state, de_strategy_evaluation, de_regression_prediction, de_classification_prediction,
) -> None:
    """Risk Manager compatibility: the placeholder shape must be identical
    no matter what the Decision Engine itself concludes."""
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(
        market_state=de_market_state, strategy_evaluation=de_strategy_evaluation,
        regression_prediction=de_regression_prediction, classification_prediction=de_classification_prediction,
        symbol="EUR_USD", timeframe="M5",
    )
    assert result.position_size.calculated_by == "RiskManager"
    assert result.position_size.status == "Pending"


# --------------------------------------------------------------------------- #
# Strategy Verdict
# --------------------------------------------------------------------------- #
def test_strategy_verdict_names_the_users_own_strategy(de_market_state, de_strategy_evaluation) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5")
    assert result.strategy_verdict.strategy_name == "trend_following"
    assert result.strategy_verdict.validation_status in ("Validated", "Conflicting", "Not Validated", "Partially Validated")


def test_strategy_verdict_historical_probability_passed_through(de_market_state, de_strategy_evaluation) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(
        market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5",
        historical_win_rate=0.58,
    )
    assert result.strategy_verdict.historical_success_probability == 0.58


def test_strategy_verdict_historical_probability_none_without_backtest_data(de_market_state, de_strategy_evaluation) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5")
    assert result.strategy_verdict.historical_success_probability is None


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def test_explainability_populated(de_market_state, de_strategy_evaluation) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5")
    assert result.reasoning.summary
    assert result.reasoning.why_buy
    assert result.reasoning.why_sell
    assert result.reasoning.why_wait
    assert result.reasoning.strategy_compliance_explanation


def test_mse_alignment_propagates_from_strategy_evaluation(de_market_state, de_strategy_evaluation) -> None:
    """StrategyEvaluation.mse_compliance must reach DecisionResult.strategy.mse_alignment
    unchanged -- this is the only place that value is threaded through the
    Decision Engine, so a dropped kwarg here silently reverts to the 100.0
    (no-gating) default instead of raising."""
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5")
    assert result.strategy.mse_alignment == de_strategy_evaluation.mse_compliance


def test_explainability_deterministic_across_repeated_calls(de_market_state, de_strategy_evaluation) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result1 = engine.decide(market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5")
    result2 = engine.decide(market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5")
    assert result1.reasoning.to_dict() == result2.reasoning.to_dict()


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_metadata_fields_populated(de_market_state, de_strategy_evaluation, de_regression_prediction, de_classification_prediction) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(
        market_state=de_market_state, strategy_evaluation=de_strategy_evaluation,
        regression_prediction=de_regression_prediction, classification_prediction=de_classification_prediction,
        symbol="EUR_USD", timeframe="M5",
    )
    meta = result.metadata
    assert meta.currency_pair == "EUR_USD"
    assert meta.timeframe == "M5"
    assert meta.timestamp
    assert meta.strategy_version == de_strategy_evaluation.strategy_version
    assert meta.decision_engine_version
    assert meta.market_structure_version
    assert meta.linear_regression_version == de_regression_prediction.model_version
    assert meta.logistic_regression_version == de_classification_prediction.model_version


def test_metadata_versions_unknown_when_models_not_supplied(de_market_state, de_strategy_evaluation) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5")
    assert result.metadata.linear_regression_version == "unknown"
    assert result.metadata.logistic_regression_version == "unknown"


def test_explicit_feature_version_overrides_inference(de_market_state, de_strategy_evaluation) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(
        market_state=de_market_state, strategy_evaluation=de_strategy_evaluation, symbol="EUR_USD", timeframe="M5",
        feature_version="9.9.9",
    )
    assert result.metadata.feature_version == "9.9.9"


# --------------------------------------------------------------------------- #
# Agentic AI compatibility -- plain, JSON-safe primitives only
# --------------------------------------------------------------------------- #
def test_to_dict_contains_only_json_safe_primitives(
    de_market_state, de_strategy_evaluation, de_regression_prediction, de_classification_prediction,
) -> None:
    engine = DecisionEngine(DecisionEngineConfig())
    result = engine.decide(
        market_state=de_market_state, strategy_evaluation=de_strategy_evaluation,
        regression_prediction=de_regression_prediction, classification_prediction=de_classification_prediction,
        symbol="EUR_USD", timeframe="M5",
    )
    serialized = json.dumps(result.to_dict())
    round_tripped = json.loads(serialized)
    assert round_tripped["recommendation"] == result.recommendation
    assert set(round_tripped) == set(result.to_dict())


def test_never_upgrades_wait_to_a_trade(de_market_state) -> None:
    """Regression guard for the spec's core safety rule: even with maximal
    model agreement, a strategy WAIT/NO_TRADE must never become BUY/SELL."""
    from decision_engine.recommendation import compute_final_recommendation

    for strategy_rec in ("WAIT", "NO_TRADE"):
        for model_net in (-1.0, -0.5, 0.0, 0.5, 1.0):
            result = compute_final_recommendation(
                strategy_recommendation=strategy_rec, strategy_direction=0, model_net=model_net,
                config=DecisionEngineConfig(),
            )
            assert result == strategy_rec
