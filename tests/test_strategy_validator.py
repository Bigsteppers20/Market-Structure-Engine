"""Tests for strategy.validator."""
from __future__ import annotations

import pytest

from strategy.config import StrategyConfig
from strategy.validator import (
    DuplicateRuleError,
    MarketStateValidationError,
    ThresholdValidationError,
    WeightValidationError,
    validate_market_state,
    validate_no_duplicate_rules,
    validate_strategy_config,
    validate_thresholds,
    validate_weights,
)


def test_validate_weights_accepts_100() -> None:
    validate_weights({"a": 60.0, "b": 40.0})  # must not raise


def test_validate_weights_rejects_non_100_sum() -> None:
    with pytest.raises(WeightValidationError):
        validate_weights({"a": 60.0, "b": 30.0})


def test_validate_weights_rejects_out_of_range() -> None:
    with pytest.raises(ThresholdValidationError):
        validate_weights({"a": 120.0, "b": -20.0})


def test_validate_weights_rejects_empty() -> None:
    with pytest.raises(WeightValidationError):
        validate_weights({})


def test_validate_no_duplicate_rules() -> None:
    validate_no_duplicate_rules(["a", "b", "c"])  # ok
    with pytest.raises(DuplicateRuleError):
        validate_no_duplicate_rules(["a", "a", "b"])


def test_validate_thresholds_range() -> None:
    validate_thresholds(70.0, 60.0)
    with pytest.raises(ThresholdValidationError):
        validate_thresholds(150.0, 60.0)
    with pytest.raises(ThresholdValidationError):
        validate_thresholds(70.0, -10.0)


def test_validate_market_state_rejects_wrong_type() -> None:
    with pytest.raises(MarketStateValidationError):
        validate_market_state("not a market state")


def test_validate_market_state_accepts_real_state(market_state) -> None:
    validate_market_state(market_state)  # must not raise


def test_validate_strategy_config_full(market_state) -> None:
    cfg = StrategyConfig(strategy_name="t", rule_weights={"trend": 60.0, "rsi": 40.0})
    validate_strategy_config(cfg)  # must not raise


def test_validate_strategy_config_rejects_enabled_rule_not_in_weights() -> None:
    cfg = StrategyConfig(
        strategy_name="t", rule_weights={"trend": 100.0}, enabled_rules={"nonexistent_rule": False},
    )
    with pytest.raises(Exception):
        validate_strategy_config(cfg)
