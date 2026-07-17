"""Weighted rule execution.

:class:`RuleEngine` runs a fixed set of ``(Rule, weight)`` pairs against one
``MarketState``, skipping disabled rules entirely and attaching each
result's configured weight plus its weighted contribution
(``weight/100 * score``). It performs no aggregation beyond that -- compliance,
confidence, and category scoring live in their own modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from market_structure import MarketState

from .rule_base import Rule, RuleResult, RuleStatus


@dataclass(slots=True)
class RuleSpec:
    """One rule plus its configured weight and enabled flag."""

    rule: Rule
    weight: float
    enabled: bool = True


class RuleEngine:
    """Evaluates every enabled rule in ``specs`` against a ``MarketState``."""

    def __init__(self, specs: List[RuleSpec]) -> None:
        names = [s.rule.name for s in specs]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate rule name(s) in RuleEngine: {dupes}")
        self.specs = specs

    def run(self, market_state: MarketState) -> List[RuleResult]:
        """Evaluate every *enabled* rule; disabled rules are skipped entirely
        (they never appear in the returned list, so they don't affect any
        downstream weight renormalization)."""
        results: List[RuleResult] = []
        for spec in self.specs:
            if not spec.enabled:
                continue
            result = spec.rule.evaluate(market_state)
            result.weight = spec.weight
            result.weighted_score = (spec.weight / 100.0) * result.score
            results.append(result)
        return results

    @property
    def enabled_specs(self) -> List[RuleSpec]:
        return [s for s in self.specs if s.enabled]


def count_by_status(results: List[RuleResult]) -> Tuple[int, int, int]:
    """Return ``(passed, failed, not_applicable)`` counts."""
    passed = sum(1 for r in results if r.status == RuleStatus.PASS)
    failed = sum(1 for r in results if r.status == RuleStatus.FAIL)
    not_applicable = sum(1 for r in results if r.status == RuleStatus.NOT_APPLICABLE)
    return passed, failed, not_applicable
