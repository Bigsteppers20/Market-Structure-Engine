"""Tests for strategy.config."""
from __future__ import annotations

import pytest

from strategy.config import StrategyConfig


def test_defaults() -> None:
    cfg = StrategyConfig(strategy_name="t")
    assert cfg.strategy_version == "1.0.0"
    assert cfg.compliance_threshold == 70.0
    assert cfg.is_enabled("anything") is True  # absent -> default enabled


def test_is_enabled_respects_explicit_false() -> None:
    cfg = StrategyConfig(strategy_name="t", rule_weights={"trend": 100.0}, enabled_rules={"trend": False})
    assert cfg.is_enabled("trend") is False


def test_to_dict_from_dict_round_trip() -> None:
    cfg = StrategyConfig(
        strategy_name="t", rule_weights={"trend": 50.0, "rsi": 50.0},
        rule_params={"rsi": {"oversold": 25.0}}, compliance_threshold=80.0,
    )
    restored = StrategyConfig.from_dict(cfg.to_dict())
    assert restored.rule_weights == cfg.rule_weights
    assert restored.rule_params == cfg.rule_params
    assert restored.compliance_threshold == 80.0


def test_from_dict_rejects_unknown_field() -> None:
    d = StrategyConfig(strategy_name="t").to_dict()
    d["bogus"] = 1
    with pytest.raises(ValueError):
        StrategyConfig.from_dict(d)


def test_json_round_trip(tmp_path) -> None:
    cfg = StrategyConfig(strategy_name="json_test", rule_weights={"trend": 100.0})
    path = cfg.to_json(tmp_path / "cfg.json")
    restored = StrategyConfig.from_json(path)
    assert restored.strategy_name == "json_test"
    assert restored.rule_weights == {"trend": 100.0}
