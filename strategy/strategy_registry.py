"""In-memory registry of strategy *classes* (not instances/configs -- see
``strategy_loader.py`` for persisting named, versioned configurations).

Mirrors the same registry pattern already used by
``ml_pipeline.label_generator.CLASSIFICATION_REGISTRY`` and
``training.registry.ModelRegistry``: register once, look up by name
anywhere, extend without modifying this file.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Type

from .strategy_base import TradingStrategy


class StrategyRegistry:
    """Name -> ``TradingStrategy`` subclass lookup."""

    def __init__(self) -> None:
        self._classes: Dict[str, Type[TradingStrategy]] = {}

    def register(self, name: str, strategy_cls: Type[TradingStrategy]) -> None:
        if not (isinstance(strategy_cls, type) and issubclass(strategy_cls, TradingStrategy)):
            raise TypeError(f"{strategy_cls!r} must be a TradingStrategy subclass.")
        self._classes[name] = strategy_cls

    def get(self, name: str) -> Type[TradingStrategy]:
        if name not in self._classes:
            raise KeyError(f"No strategy class registered under {name!r}. Registered: {sorted(self._classes)}")
        return self._classes[name]

    def list_classes(self) -> List[str]:
        return sorted(self._classes)

    def unregister(self, name: str) -> None:
        self._classes.pop(name, None)


_default_registry: Optional[StrategyRegistry] = None


def default_registry() -> StrategyRegistry:
    """The process-wide registry, pre-populated with the 5 built-in strategies
    from the top-level ``strategies/`` package (imported lazily here to avoid
    a circular import -- ``strategies/*.py`` imports from ``strategy/``)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = StrategyRegistry()
        _register_builtin_strategies(_default_registry)
    return _default_registry


def _register_builtin_strategies(registry: StrategyRegistry) -> None:
    from strategies.ict_strategy import IctStrategy
    from strategies.london_breakout import LondonBreakoutStrategy
    from strategies.scalping_strategy import ScalpingStrategy
    from strategies.swing_strategy import SwingStrategy
    from strategies.trend_following import TrendFollowingStrategy

    registry.register("ict", IctStrategy)
    registry.register("trend_following", TrendFollowingStrategy)
    registry.register("london_breakout", LondonBreakoutStrategy)
    registry.register("swing", SwingStrategy)
    registry.register("scalping", ScalpingStrategy)
