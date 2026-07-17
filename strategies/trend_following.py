"""Trend-following strategy: ride established directional moves confirmed by
EMA stacking, momentum, and MACD.
"""
from __future__ import annotations

from typing import Dict

from strategy.config import StrategyConfig
from strategy.rule_base import Rule, build_all_rules
from strategy.strategy_base import TradingStrategy

DEFAULT_WEIGHTS: Dict[str, float] = {
    "trend": 30.0,
    "ema_alignment": 20.0,
    "momentum": 15.0,
    "macd": 15.0,
    "swing_structure": 10.0,
    "atr": 5.0,
    "session": 5.0,
}


def default_config(strategy_version: str = "1.0.0") -> StrategyConfig:
    return StrategyConfig(
        strategy_name="trend_following",
        strategy_version=strategy_version,
        rule_weights=dict(DEFAULT_WEIGHTS),
        rule_params={"trend": {"min_strength": 0.4}},
        compliance_threshold=65.0,
        confidence_threshold=60.0,
    )


class TrendFollowingStrategy(TradingStrategy):
    """Directional-continuation strategy: heavy weight on trend strength,
    EMA stack alignment, and momentum/MACD confirmation."""

    def build_rules(self) -> Dict[str, Rule]:
        return build_all_rules(self.config.rule_params)
