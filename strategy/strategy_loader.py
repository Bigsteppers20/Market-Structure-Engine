"""Save/load named, versioned ``StrategyConfig`` instances to disk, and build
ready-to-use ``TradingStrategy`` instances from them.

File-based, same convention as ``training.registry.ModelRegistry``: one JSON
per ``name@version`` plus an index for fast listing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from training.utils import ensure_dir

from .config import StrategyConfig
from .strategy_base import TradingStrategy
from .strategy_registry import StrategyRegistry, default_registry
from .validator import validate_strategy_config


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in value)


class StrategyLoader:
    """Persists :class:`StrategyConfig` objects and builds strategies from them."""

    def __init__(self, root: str | Path, registry: Optional[StrategyRegistry] = None) -> None:
        self.root = ensure_dir(Path(root) / "strategy_configs")
        self.registry = registry or default_registry()
        self._index_path = self.root / "index.json"

    # ------------------------------------------------------------------ #
    def save(self, config: StrategyConfig) -> Path:
        """Validate and persist a configuration under its name+version."""
        validate_strategy_config(config)
        path = self.root / f"{_safe(config.strategy_name)}__{_safe(config.strategy_version)}.json"
        config.to_json(path)
        self._update_index(config)
        return path

    def load_config(self, name: str, version: Optional[str] = None) -> StrategyConfig:
        """Load a saved configuration; the most-recently-saved version if
        ``version`` is omitted."""
        index = self._load_index()
        versions = index.get(name)
        if not versions:
            raise KeyError(f"No saved configuration named {name!r}. Known: {sorted(index)}")
        version = version or versions[-1]
        if version not in versions:
            raise KeyError(f"Configuration {name!r} has no version {version!r}. Known: {versions}")
        path = self.root / f"{_safe(name)}__{_safe(version)}.json"
        return StrategyConfig.from_json(path)

    def list_saved(self) -> List[str]:
        return sorted(self._load_index())

    def list_versions(self, name: str) -> List[str]:
        return list(self._load_index().get(name, []))

    # ------------------------------------------------------------------ #
    def build(
        self, strategy_class_name: str, *, config: Optional[StrategyConfig] = None,
        config_name: Optional[str] = None, config_version: Optional[str] = None,
    ) -> TradingStrategy:
        """Instantiate a registered strategy class.

        Pass ``config`` directly for an in-memory configuration, or
        ``config_name``/``config_version`` to load a previously
        :meth:`save`-d one (defaulting ``config_name`` to
        ``strategy_class_name`` when omitted).
        """
        strategy_cls = self.registry.get(strategy_class_name)
        cfg = config or self.load_config(config_name or strategy_class_name, config_version)
        return strategy_cls(cfg)

    # ------------------------------------------------------------------ #
    def _load_index(self) -> Dict[str, List[str]]:
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def _update_index(self, config: StrategyConfig) -> None:
        index = self._load_index()
        versions = index.setdefault(config.strategy_name, [])
        if config.strategy_version in versions:
            versions.remove(config.strategy_version)
        versions.append(config.strategy_version)
        self._index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
