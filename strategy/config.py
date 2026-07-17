"""User-facing strategy configuration.

Everything a user can tune without writing code: which rules are active,
their weights, per-rule parameters, and the compliance/confidence
thresholds that separate BUY/SELL from WAIT. See ``strategy_loader.py`` for
saving/loading named, versioned configurations to disk.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class StrategyConfig:
    """Complete, user-editable configuration for one strategy instance.

    Attributes
    ----------
    strategy_name:
        Unique name this configuration is saved/loaded under.
    strategy_version, rule_version, configuration_version:
        See :mod:`strategy.strategy_version`.
    rule_weights:
        ``rule_name -> weight`` (percentage points). Must sum to 100 across
        the *full* set (validated by :func:`strategy.validator.validate_weights`) --
        this is the strategy's authored weighting scheme.
    enabled_rules:
        ``rule_name -> bool``. Disabling a rule removes it from evaluation
        entirely (not merely forcing NOT_APPLICABLE) -- compliance/scoring
        renormalize among whatever actually ran, exactly as they already do
        for NOT_APPLICABLE rules. Rules absent from this dict default to
        enabled. Weights are **not** re-validated to sum to 100 after
        disabling -- that would make "temporarily disable one rule" require
        rebalancing everything else.
    rule_params:
        ``rule_name -> {constructor kwarg -> value}``, forwarded when a rule
        instance is built (e.g. ``{"rsi": {"oversold": 25.0}}``).
    compliance_threshold, confidence_threshold:
        Both in [0, 100]. A BUY/SELL recommendation requires compliance and
        confidence to each meet their threshold; otherwise WAIT.
    symbol, timeframe:
        Declared applicability (``"*"`` = any). Purely descriptive here --
        enforcement, if any, is the caller's responsibility.
    overall_score_weights:
        Blend weights for :func:`strategy.scoring.compute_scores`'s
        ``overall_score`` (must be the keys ``"technical"``,
        ``"market_quality"``, ``"risk"``).
    """

    strategy_name: str
    strategy_version: str = "1.0.0"
    rule_version: str = "1.0.0"
    configuration_version: str = "1.0.0"
    rule_weights: Dict[str, float] = field(default_factory=dict)
    enabled_rules: Dict[str, bool] = field(default_factory=dict)
    rule_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    compliance_threshold: float = 70.0
    confidence_threshold: float = 60.0
    symbol: str = "*"
    timeframe: str = "*"
    overall_score_weights: Dict[str, float] = field(
        default_factory=lambda: {"technical": 0.5, "market_quality": 0.3, "risk": 0.2}
    )

    def is_enabled(self, rule_name: str) -> bool:
        return self.enabled_rules.get(rule_name, True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown StrategyConfig field(s): {sorted(unknown)}")
        return cls(**d)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "StrategyConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
