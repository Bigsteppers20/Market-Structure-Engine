"""London breakout strategy: session-gated volatility expansion trades
around the London open, confirmed by a fresh break of structure.
"""
from __future__ import annotations

from typing import Dict

from strategy.config import StrategyConfig
from strategy.rule_base import Rule, build_all_rules
from strategy.strategy_base import TradingStrategy

DEFAULT_WEIGHTS: Dict[str, float] = {
    "session": 25.0,
    "break_of_structure": 20.0,
    "volatility": 15.0,
    "spread": 15.0,
    "atr": 10.0,
    "liquidity_sweep": 10.0,
    "trend": 5.0,
}


def default_config(strategy_version: str = "1.0.0") -> StrategyConfig:
    return StrategyConfig(
        strategy_name="london_breakout",
        strategy_version=strategy_version,
        rule_weights=dict(DEFAULT_WEIGHTS),
        rule_params={"session": {"preferred_sessions": ("is_london",)}},
        compliance_threshold=70.0,
        confidence_threshold=55.0,
    )


class LondonBreakoutStrategy(TradingStrategy):
    """Session-gated breakout strategy: only meaningfully compliant during
    the London session, weighted toward volatility expansion + a fresh BOS."""

    def build_rules(self) -> Dict[str, Rule]:
        return build_all_rules(self.config.rule_params)
