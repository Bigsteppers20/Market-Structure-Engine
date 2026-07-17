"""Tests for the 5 concrete strategies in the top-level strategies/ package."""
from __future__ import annotations

import pytest

from strategies.ict_strategy import IctStrategy, default_config as ict_config
from strategies.london_breakout import LondonBreakoutStrategy, default_config as london_config
from strategies.scalping_strategy import ScalpingStrategy, default_config as scalping_config
from strategies.swing_strategy import SwingStrategy, default_config as swing_config
from strategies.trend_following import TrendFollowingStrategy, default_config as trend_config
from strategy.validator import validate_strategy_config


@pytest.mark.parametrize("cls,config_fn", [
    (IctStrategy, ict_config),
    (TrendFollowingStrategy, trend_config),
    (LondonBreakoutStrategy, london_config),
    (SwingStrategy, swing_config),
    (ScalpingStrategy, scalping_config),
])
def test_default_config_weights_sum_to_100(cls, config_fn) -> None:
    cfg = config_fn()
    total = sum(cfg.rule_weights.values())
    assert total == pytest.approx(100.0, abs=0.01)
    validate_strategy_config(cfg)  # must not raise


@pytest.mark.parametrize("cls,config_fn", [
    (IctStrategy, ict_config),
    (TrendFollowingStrategy, trend_config),
    (LondonBreakoutStrategy, london_config),
    (SwingStrategy, swing_config),
    (ScalpingStrategy, scalping_config),
])
def test_strategy_evaluates_real_market_state(cls, config_fn, market_state) -> None:
    strategy = cls(config_fn())
    evaluation = strategy.evaluate(market_state, symbol="EUR_USD", timeframe="M5")
    assert evaluation.strategy_name == cfg_name(config_fn)
    assert len(evaluation.rule_results) == len(config_fn().rule_weights)
    assert 0.0 <= evaluation.strategy_compliance <= 100.0
    assert 0.0 <= evaluation.strategy_confidence <= 100.0


def cfg_name(config_fn) -> str:
    return config_fn().strategy_name


def test_ict_strategy_matches_spec_example_weights() -> None:
    cfg = ict_config()
    assert cfg.rule_weights == {
        "trend": 20.0, "break_of_structure": 15.0, "liquidity_sweep": 15.0,
        "order_block": 15.0, "ema_alignment": 10.0, "rsi": 10.0,
        "macd": 5.0, "session": 5.0, "spread": 5.0,
    }


def test_strategies_produce_different_evaluations_on_same_state(market_state) -> None:
    """Different weightings should generally disagree at least somewhat --
    a sanity check that strategies aren't accidentally identical."""
    ict_eval = IctStrategy(ict_config()).evaluate(market_state)
    swing_eval = SwingStrategy(swing_config()).evaluate(market_state)
    # They may occasionally agree on bias, but the rule sets differ so the
    # score composition must differ.
    assert ict_eval.technical_score != swing_eval.technical_score or \
           ict_eval.market_quality_score != swing_eval.market_quality_score


def test_all_five_strategies_registered_in_default_registry() -> None:
    from strategy.strategy_registry import default_registry
    reg = default_registry()
    assert reg.get("ict") is IctStrategy
    assert reg.get("trend_following") is TrendFollowingStrategy
    assert reg.get("london_breakout") is LondonBreakoutStrategy
    assert reg.get("swing") is SwingStrategy
    assert reg.get("scalping") is ScalpingStrategy
