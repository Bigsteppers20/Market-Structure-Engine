"""Tests for decision_engine.trade_plan_builder -- TRADE PLAN section.

Uses a lightweight duck-typed fake MarketState (only the fields
trade_plan_builder actually reads: price_action.current_close,
indicators["atr"], indicator_validity["atr"]) rather than running the full
Market Structure Engine, for fast, isolated unit tests of the geometry.
"""
from __future__ import annotations

import pytest

from decision_engine.config import DecisionEngineConfig
from decision_engine.decision_result import LinearRegressionAnalysis
from decision_engine.trade_plan_builder import build_trade_plan


class _FakePriceAction:
    def __init__(self, current_close: float) -> None:
        self.current_close = current_close


class _FakeMarketState:
    def __init__(self, current_close: float = 1.1000, atr: float = 0.0010, atr_valid: float = 1.0) -> None:
        self.price_action = _FakePriceAction(current_close)
        self.indicators = {"atr": atr}
        self.indicator_validity = {"atr": atr_valid}


def _no_regression() -> LinearRegressionAnalysis:
    return LinearRegressionAnalysis(available=False)


def test_none_direction_when_recommendation_is_wait() -> None:
    plan = build_trade_plan(
        final_recommendation="WAIT", market_state=_FakeMarketState(), linear_regression=_no_regression(),
        consensus_score=50.0, opportunity_score=50.0, prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.direction == "NONE"
    assert plan.entry_price is None
    assert plan.stop_loss is None
    assert plan.target_feasibility == 0.0


def test_none_direction_when_recommendation_is_no_trade() -> None:
    plan = build_trade_plan(
        final_recommendation="NO_TRADE", market_state=_FakeMarketState(), linear_regression=_no_regression(),
        consensus_score=50.0, opportunity_score=50.0, prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.direction == "NONE"


def test_buy_plan_places_stop_below_and_targets_above_entry() -> None:
    plan = build_trade_plan(
        final_recommendation="BUY", market_state=_FakeMarketState(current_close=1.1000, atr=0.0010),
        linear_regression=_no_regression(), consensus_score=70.0, opportunity_score=70.0,
        prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.direction == "BUY"
    assert plan.entry_price == 1.1000
    assert plan.stop_loss < plan.entry_price
    assert plan.take_profit_1 > plan.entry_price
    assert plan.take_profit_2 > plan.take_profit_1
    assert plan.take_profit_3 > plan.take_profit_2


def test_sell_plan_places_stop_above_and_targets_below_entry() -> None:
    plan = build_trade_plan(
        final_recommendation="SELL", market_state=_FakeMarketState(current_close=1.1000, atr=0.0010),
        linear_regression=_no_regression(), consensus_score=70.0, opportunity_score=70.0,
        prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.direction == "SELL"
    assert plan.stop_loss > plan.entry_price
    assert plan.take_profit_1 < plan.entry_price
    assert plan.take_profit_2 < plan.take_profit_1
    assert plan.take_profit_3 < plan.take_profit_2


def test_risk_reward_ratio_matches_tp2_multiple() -> None:
    config = DecisionEngineConfig(take_profit_r_multiples=(1.0, 2.5, 4.0))
    plan = build_trade_plan(
        final_recommendation="BUY", market_state=_FakeMarketState(), linear_regression=_no_regression(),
        consensus_score=70.0, opportunity_score=70.0, prediction_horizon=5, timeframe="M5", config=config,
    )
    assert plan.risk_reward_ratio == pytest.approx(2.5)


def test_stop_distance_scales_with_atr_multiple() -> None:
    config_tight = DecisionEngineConfig(stop_atr_multiple=1.0)
    config_wide = DecisionEngineConfig(stop_atr_multiple=3.0)
    market_state = _FakeMarketState(current_close=1.1000, atr=0.0010)
    plan_tight = build_trade_plan(
        final_recommendation="BUY", market_state=market_state, linear_regression=_no_regression(),
        consensus_score=70.0, opportunity_score=70.0, prediction_horizon=5, timeframe="M5", config=config_tight,
    )
    plan_wide = build_trade_plan(
        final_recommendation="BUY", market_state=market_state, linear_regression=_no_regression(),
        consensus_score=70.0, opportunity_score=70.0, prediction_horizon=5, timeframe="M5", config=config_wide,
    )
    tight_distance = plan_tight.entry_price - plan_tight.stop_loss
    wide_distance = plan_wide.entry_price - plan_wide.stop_loss
    assert wide_distance == pytest.approx(tight_distance * 3.0)


def test_invalid_atr_yields_entry_only_no_stop_or_targets() -> None:
    plan = build_trade_plan(
        final_recommendation="BUY", market_state=_FakeMarketState(atr_valid=0.0), linear_regression=_no_regression(),
        consensus_score=70.0, opportunity_score=70.0, prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.direction == "BUY"
    assert plan.entry_price is not None
    assert plan.stop_loss is None
    assert plan.target_feasibility == 0.0


def test_zero_atr_treated_as_invalid() -> None:
    plan = build_trade_plan(
        final_recommendation="BUY", market_state=_FakeMarketState(atr=0.0), linear_regression=_no_regression(),
        consensus_score=70.0, opportunity_score=70.0, prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.stop_loss is None


def test_expected_holding_time_formatted_from_horizon() -> None:
    plan = build_trade_plan(
        final_recommendation="BUY", market_state=_FakeMarketState(), linear_regression=_no_regression(),
        consensus_score=70.0, opportunity_score=70.0, prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert "5" in plan.expected_holding_time and "M5" in plan.expected_holding_time


def test_expected_holding_time_unknown_without_horizon() -> None:
    plan = build_trade_plan(
        final_recommendation="BUY", market_state=_FakeMarketState(), linear_regression=_no_regression(),
        consensus_score=70.0, opportunity_score=70.0, prediction_horizon=None, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.expected_holding_time == "Unknown"


def test_regression_pip_movement_used_when_direction_agrees() -> None:
    lr = LinearRegressionAnalysis(available=True, expected_pip_movement=25.0, expected_MFE=0.003, expected_MAE=0.001)
    plan = build_trade_plan(
        final_recommendation="BUY", market_state=_FakeMarketState(current_close=1.1000, atr=0.0010),
        linear_regression=lr, consensus_score=70.0, opportunity_score=70.0,
        prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.expected_pip_gain == pytest.approx(25.0)


def test_regression_pip_movement_ignored_when_direction_disagrees() -> None:
    lr = LinearRegressionAnalysis(available=True, expected_pip_movement=-25.0)
    plan = build_trade_plan(
        final_recommendation="BUY", market_state=_FakeMarketState(current_close=1.1000, atr=0.0010),
        linear_regression=lr, consensus_score=70.0, opportunity_score=70.0,
        prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.expected_pip_gain != pytest.approx(-25.0)
    assert plan.expected_pip_gain > 0


def test_expected_maximum_drawdown_uses_regression_mae_when_available() -> None:
    lr = LinearRegressionAnalysis(available=True, expected_MAE=0.0020)
    plan = build_trade_plan(
        final_recommendation="BUY", market_state=_FakeMarketState(), linear_regression=lr,
        consensus_score=70.0, opportunity_score=70.0, prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.expected_maximum_drawdown == pytest.approx(0.0020 / 0.0001)


def test_target_feasibility_high_when_mfe_covers_target() -> None:
    lr = LinearRegressionAnalysis(available=True, expected_MFE=1.0)  # far more than any TP distance
    plan = build_trade_plan(
        final_recommendation="BUY", market_state=_FakeMarketState(current_close=1.1000, atr=0.0010),
        linear_regression=lr, consensus_score=70.0, opportunity_score=70.0,
        prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.target_feasibility == 100.0


def test_target_feasibility_falls_back_to_consensus_without_mfe() -> None:
    plan = build_trade_plan(
        final_recommendation="BUY", market_state=_FakeMarketState(), linear_regression=_no_regression(),
        consensus_score=42.0, opportunity_score=70.0, prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.target_feasibility == pytest.approx(42.0)


def test_trade_quality_score_in_valid_range() -> None:
    plan = build_trade_plan(
        final_recommendation="BUY", market_state=_FakeMarketState(), linear_regression=_no_regression(),
        consensus_score=70.0, opportunity_score=70.0, prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert 0.0 <= plan.trade_quality_score <= 100.0


def test_no_trade_still_produces_a_trade_quality_score_for_context() -> None:
    plan = build_trade_plan(
        final_recommendation="WAIT", market_state=_FakeMarketState(), linear_regression=_no_regression(),
        consensus_score=70.0, opportunity_score=70.0, prediction_horizon=5, timeframe="M5", config=DecisionEngineConfig(),
    )
    assert plan.trade_quality_score > 0.0
