"""Configuration and input validation, with descriptive exceptions.

Checked before every training/definition change and before every live
evaluation: rule weights, duplicate rules, threshold ranges, and that the
object handed to ``evaluate()`` is actually a usable ``MarketState``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from market_structure import MarketState

if TYPE_CHECKING:
    from .config import StrategyConfig

WEIGHT_SUM_TOLERANCE = 0.01


class StrategyValidationError(ValueError):
    """Base class for every validation failure in this module."""


class WeightValidationError(StrategyValidationError):
    """Rule weights don't sum to 100% (within tolerance)."""


class DuplicateRuleError(StrategyValidationError):
    """The same rule name appears more than once in a configuration."""


class ThresholdValidationError(StrategyValidationError):
    """A compliance/confidence threshold (or a rule weight) is out of range."""


class MarketStateValidationError(StrategyValidationError):
    """The object passed to ``evaluate()`` is not a usable ``MarketState``."""


def validate_weights(rule_weights: Dict[str, float]) -> None:
    """Every configured weight must be in [0, 100] and the set must sum to 100%."""
    if not rule_weights:
        raise WeightValidationError("rule_weights is empty -- a strategy needs at least one rule.")
    for name, weight in rule_weights.items():
        if not (0.0 <= weight <= 100.0):
            raise ThresholdValidationError(
                f"Rule {name!r} has weight {weight!r}, outside the valid range [0, 100]."
            )
    total = sum(rule_weights.values())
    if abs(total - 100.0) > WEIGHT_SUM_TOLERANCE:
        raise WeightValidationError(
            f"Rule weights sum to {total:.4f}, expected 100.0 (+/-{WEIGHT_SUM_TOLERANCE}). "
            f"Weights: {rule_weights}"
        )


def validate_no_duplicate_rules(rule_names: List[str]) -> None:
    if len(rule_names) != len(set(rule_names)):
        dupes = sorted({n for n in rule_names if rule_names.count(n) > 1})
        raise DuplicateRuleError(f"Duplicate rule name(s): {dupes}")


def validate_thresholds(compliance_threshold: float, confidence_threshold: float) -> None:
    for label, value in (("compliance_threshold", compliance_threshold),
                          ("confidence_threshold", confidence_threshold)):
        if not (0.0 <= value <= 100.0):
            raise ThresholdValidationError(f"{label}={value!r} must be in [0, 100].")


def validate_market_state(market_state: MarketState) -> None:
    if not isinstance(market_state, MarketState):
        raise MarketStateValidationError(
            f"Expected a market_structure.MarketState instance, got {type(market_state)!r}."
        )
    if market_state.n_candles <= 0:
        raise MarketStateValidationError("MarketState.n_candles must be > 0 -- was analyze() called?")
    if market_state.trend is None:
        raise MarketStateValidationError(
            "MarketState.trend is None -- this MarketState was not produced by "
            "MarketStructureEngine.analyze() (or analyze() failed)."
        )


def validate_strategy_config(config: "StrategyConfig") -> None:
    """Full validation of a :class:`~strategy.config.StrategyConfig`."""
    validate_no_duplicate_rules(list(config.rule_weights.keys()))
    validate_weights(config.rule_weights)
    validate_thresholds(config.compliance_threshold, config.confidence_threshold)
    unknown_enabled = set(config.enabled_rules) - set(config.rule_weights)
    if unknown_enabled:
        raise StrategyValidationError(
            f"enabled_rules references rule(s) not present in rule_weights: {sorted(unknown_enabled)}"
        )
