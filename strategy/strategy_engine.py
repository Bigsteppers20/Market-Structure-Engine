"""The production entry point: ``MarketState`` in, ``StrategyEvaluation`` out.

Live operation is exactly this, per candle, entirely in memory::

    market_state = mse_engine.analyze()          # Market Structure Engine
    evaluation = strategy_engine.evaluate(        # Strategy Engine
        market_state, strategy_name="ict", symbol="EUR_USD", timeframe="M5",
    )

No CSV files, no offline batch step, no broker call, no raw candle access --
``StrategyEngine`` only ever touches the already-configured
``TradingStrategy`` instances registered with it and the ``MarketState``
handed to :meth:`evaluate`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from market_structure import MarketState

from .strategy_base import StrategyEvaluation, TradingStrategy
from .strategy_registry import StrategyRegistry, default_registry


class StrategyEngine:
    """Holds a set of ready-to-use strategy instances and evaluates them."""

    def __init__(self, registry: Optional[StrategyRegistry] = None) -> None:
        self.registry = registry or default_registry()
        self._strategies: Dict[str, TradingStrategy] = {}

    def register_strategy(self, strategy: TradingStrategy) -> None:
        """Add a configured strategy instance, keyed by its own ``.name``."""
        self._strategies[strategy.name] = strategy

    def unregister_strategy(self, name: str) -> None:
        self._strategies.pop(name, None)

    def list_strategies(self) -> List[str]:
        return sorted(self._strategies)

    def get_strategy(self, name: str) -> TradingStrategy:
        if name not in self._strategies:
            raise KeyError(f"No strategy registered as {name!r} in this engine. Registered: {self.list_strategies()}")
        return self._strategies[name]

    # ------------------------------------------------------------------ #
    def evaluate(
        self, market_state: MarketState, strategy_name: str,
        symbol: str = "UNKNOWN", timeframe: str = "UNKNOWN",
    ) -> StrategyEvaluation:
        """Evaluate one registered strategy against the current ``MarketState``."""
        return self.get_strategy(strategy_name).evaluate(market_state, symbol=symbol, timeframe=timeframe)

    def evaluate_all(
        self, market_state: MarketState, symbol: str = "UNKNOWN", timeframe: str = "UNKNOWN",
    ) -> Dict[str, StrategyEvaluation]:
        """Evaluate every registered strategy against the same ``MarketState`` --
        useful for a downstream Decision Engine / Agentic AI comparing strategies."""
        return {
            name: strategy.evaluate(market_state, symbol=symbol, timeframe=timeframe)
            for name, strategy in self._strategies.items()
        }
