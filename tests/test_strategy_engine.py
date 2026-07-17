"""Tests for strategy.strategy_engine -- the live, in-memory entry point."""
from __future__ import annotations

import pytest

from strategy.config import StrategyConfig
from strategy.strategy_engine import StrategyEngine
from strategy.strategy_registry import StrategyRegistry
from strategy.strategy_base import TradingStrategy
from strategy.rule_base import build_all_rules


class _DummyStrategy(TradingStrategy):
    def build_rules(self):
        return build_all_rules(self.config.rule_params)


def _dummy(name="dummy"):
    return _DummyStrategy(StrategyConfig(strategy_name=name, rule_weights={"trend": 60.0, "rsi": 40.0}))


def test_register_and_evaluate(market_state) -> None:
    engine = StrategyEngine(StrategyRegistry())
    engine.register_strategy(_dummy())
    evaluation = engine.evaluate(market_state, "dummy", symbol="EUR_USD", timeframe="M5")
    assert evaluation.strategy_name == "dummy"
    assert evaluation.symbol == "EUR_USD"


def test_evaluate_unknown_strategy_raises(market_state) -> None:
    engine = StrategyEngine(StrategyRegistry())
    with pytest.raises(KeyError):
        engine.evaluate(market_state, "does_not_exist")


def test_list_and_unregister_strategies() -> None:
    engine = StrategyEngine(StrategyRegistry())
    engine.register_strategy(_dummy("a"))
    engine.register_strategy(_dummy("b"))
    assert engine.list_strategies() == ["a", "b"]
    engine.unregister_strategy("a")
    assert engine.list_strategies() == ["b"]


def test_evaluate_all_runs_every_registered_strategy(market_state) -> None:
    engine = StrategyEngine(StrategyRegistry())
    engine.register_strategy(_dummy("a"))
    engine.register_strategy(_dummy("b"))
    results = engine.evaluate_all(market_state, symbol="EUR_USD", timeframe="M5")
    assert set(results) == {"a", "b"}
    assert all(ev.symbol == "EUR_USD" for ev in results.values())


def test_get_strategy(market_state) -> None:
    engine = StrategyEngine(StrategyRegistry())
    strategy = _dummy()
    engine.register_strategy(strategy)
    assert engine.get_strategy("dummy") is strategy
    with pytest.raises(KeyError):
        engine.get_strategy("nope")
