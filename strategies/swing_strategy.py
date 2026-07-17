"""Swing strategy: multi-day structure trades off support/resistance zones
and order blocks, filtered by trend and RSI.
"""
from __future__ import annotations

from typing import Dict

from strategy.config import StrategyConfig
from strategy.rule_base import Rule, build_all_rules
from strategy.strategy_base import TradingStrategy

DEFAULT_WEIGHTS: Dict[str, float] = {
    "swing_structure": 20.0,
    "support": 15.0,
    "resistance": 15.0,
    "trend": 15.0,
    "rsi": 10.0,
    "order_block": 10.0,
    "fair_value_gap": 10.0,
    "volatility": 5.0,
}


def default_config(strategy_version: str = "1.0.0") -> StrategyConfig:
    return StrategyConfig(
        strategy_name="swing",
        strategy_version=strategy_version,
        rule_weights=dict(DEFAULT_WEIGHTS),
        rule_params={"support": {"max_distance_atr": 2.5}, "resistance": {"max_distance_atr": 2.5}},
        compliance_threshold=65.0,
        confidence_threshold=55.0,
    )


class SwingStrategy(TradingStrategy):
    """Longer-horizon strategy: leans on zone quality (support/resistance)
    and structural context rather than short-term momentum/session timing."""

    def build_rules(self) -> Dict[str, Rule]:
        return build_all_rules(self.config.rule_params)
