"""Tests for strategy.strategy_registry and strategy.strategy_loader."""
from __future__ import annotations

import pytest

from strategy.config import StrategyConfig
from strategy.strategy_base import TradingStrategy
from strategy.strategy_loader import StrategyLoader
from strategy.strategy_registry import StrategyRegistry, default_registry
from strategy.rule_base import Rule, build_all_rules


class _DummyStrategy(TradingStrategy):
    def build_rules(self):
        return build_all_rules(self.config.rule_params)


def test_register_and_get() -> None:
    reg = StrategyRegistry()
    reg.register("dummy", _DummyStrategy)
    assert reg.get("dummy") is _DummyStrategy


def test_get_unknown_raises() -> None:
    reg = StrategyRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_register_rejects_non_trading_strategy() -> None:
    reg = StrategyRegistry()
    with pytest.raises(TypeError):
        reg.register("bad", object)


def test_list_classes_and_unregister() -> None:
    reg = StrategyRegistry()
    reg.register("dummy", _DummyStrategy)
    assert reg.list_classes() == ["dummy"]
    reg.unregister("dummy")
    assert reg.list_classes() == []


def test_default_registry_has_5_builtin_strategies() -> None:
    reg = default_registry()
    assert set(reg.list_classes()) == {"ict", "trend_following", "london_breakout", "swing", "scalping"}


def test_loader_save_and_load_config(tmp_path) -> None:
    reg = StrategyRegistry()
    reg.register("dummy", _DummyStrategy)
    loader = StrategyLoader(tmp_path, reg)
    cfg = StrategyConfig(strategy_name="dummy", rule_weights={"trend": 100.0})
    loader.save(cfg)
    loaded = loader.load_config("dummy")
    assert loaded.rule_weights == {"trend": 100.0}


def test_loader_load_missing_raises(tmp_path) -> None:
    loader = StrategyLoader(tmp_path, StrategyRegistry())
    with pytest.raises(KeyError):
        loader.load_config("missing")


def test_loader_multiple_versions(tmp_path) -> None:
    reg = StrategyRegistry()
    reg.register("dummy", _DummyStrategy)
    loader = StrategyLoader(tmp_path, reg)
    loader.save(StrategyConfig(strategy_name="dummy", strategy_version="1.0.0", rule_weights={"trend": 100.0}))
    loader.save(StrategyConfig(strategy_name="dummy", strategy_version="2.0.0", rule_weights={"rsi": 100.0}))
    assert loader.list_versions("dummy") == ["1.0.0", "2.0.0"]
    assert loader.load_config("dummy").strategy_version == "2.0.0"  # latest by default
    assert loader.load_config("dummy", version="1.0.0").rule_weights == {"trend": 100.0}


def test_loader_build_instantiates_strategy(tmp_path) -> None:
    reg = StrategyRegistry()
    reg.register("dummy", _DummyStrategy)
    loader = StrategyLoader(tmp_path, reg)
    cfg = StrategyConfig(strategy_name="dummy", rule_weights={"trend": 100.0})
    loader.save(cfg)
    strategy = loader.build("dummy")
    assert isinstance(strategy, _DummyStrategy)
    assert strategy.name == "dummy"


def test_loader_build_with_inline_config_skips_disk(tmp_path) -> None:
    reg = StrategyRegistry()
    reg.register("dummy", _DummyStrategy)
    loader = StrategyLoader(tmp_path, reg)
    cfg = StrategyConfig(strategy_name="in_memory", rule_weights={"rsi": 100.0})
    strategy = loader.build("dummy", config=cfg)
    assert strategy.name == "in_memory"
