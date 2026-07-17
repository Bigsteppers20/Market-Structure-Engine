"""ICT-style strategy: trend + Break of Structure + liquidity sweeps + order
blocks, exactly the weighting given as the worked example in the Strategy
Engine spec.
"""
from __future__ import annotations

from typing import Dict

from strategy.config import StrategyConfig
from strategy.rule_base import Rule, build_all_rules
from strategy.strategy_base import TradingStrategy

#: The exact example weighting from the spec (sums to 100).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "trend": 20.0,
    "break_of_structure": 15.0,
    "liquidity_sweep": 15.0,
    "order_block": 15.0,
    "ema_alignment": 10.0,
    "rsi": 10.0,
    "macd": 5.0,
    "session": 5.0,
    "spread": 5.0,
}


def default_config(strategy_version: str = "1.0.0") -> StrategyConfig:
    return StrategyConfig(
        strategy_name="ict",
        strategy_version=strategy_version,
        rule_weights=dict(DEFAULT_WEIGHTS),
        compliance_threshold=70.0,
        confidence_threshold=60.0,
    )


class IctStrategy(TradingStrategy):
    """Inner Circle Trader style: liquidity sweep -> BOS -> order block/FVG
    entry, filtered by trend, EMA alignment, and session."""

    def build_rules(self) -> Dict[str, Rule]:
        return build_all_rules(self.config.rule_params)
