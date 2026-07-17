"""Scalping strategy: very short-horizon trades gated almost entirely by
execution quality (spread, volatility) plus immediate momentum.
"""
from __future__ import annotations

from typing import Dict

from strategy.config import StrategyConfig
from strategy.rule_base import Rule, build_all_rules
from strategy.strategy_base import TradingStrategy

DEFAULT_WEIGHTS: Dict[str, float] = {
    "spread": 25.0,
    "volatility": 20.0,
    "momentum": 15.0,
    "session": 15.0,
    "rsi": 10.0,
    "volume": 10.0,
    "atr": 5.0,
}


def default_config(strategy_version: str = "1.0.0") -> StrategyConfig:
    return StrategyConfig(
        strategy_name="scalping",
        strategy_version=strategy_version,
        rule_weights=dict(DEFAULT_WEIGHTS),
        rule_params={"spread": {"max_percentile": 0.6}},
        compliance_threshold=75.0,
        confidence_threshold=65.0,
    )


class ScalpingStrategy(TradingStrategy):
    """Execution-quality-first strategy: a wide spread or expanding
    volatility disqualifies a setup regardless of how directional it looks."""

    def build_rules(self) -> Dict[str, Rule]:
        return build_all_rules(self.config.rule_params)
