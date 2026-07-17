"""Tests for strategy.rule_engine."""
from __future__ import annotations

import pytest

from strategy.rule_base import Rule, RuleResult, RuleStatus
from strategy.rule_engine import RuleEngine, RuleSpec, count_by_status


class _ConstRule(Rule):
    category = "technical"

    def __init__(self, name: str, status: RuleStatus, score: float):
        self.name = name
        self._status = status
        self._score = score

    def evaluate(self, market_state) -> RuleResult:
        return RuleResult(rule_name=self.name, category=self.category, status=self._status,
                           score=self._score, confidence=80.0, reason="const")


def test_run_attaches_weight_and_weighted_score() -> None:
    specs = [RuleSpec(rule=_ConstRule("a", RuleStatus.PASS, 80.0), weight=60.0),
             RuleSpec(rule=_ConstRule("b", RuleStatus.FAIL, 20.0), weight=40.0)]
    results = RuleEngine(specs).run(market_state=None)
    a = next(r for r in results if r.rule_name == "a")
    assert a.weight == 60.0
    assert a.weighted_score == pytest.approx(0.6 * 80.0)


def test_disabled_rule_never_appears_in_results() -> None:
    specs = [RuleSpec(rule=_ConstRule("a", RuleStatus.PASS, 80.0), weight=50.0),
             RuleSpec(rule=_ConstRule("b", RuleStatus.PASS, 90.0), weight=50.0, enabled=False)]
    results = RuleEngine(specs).run(market_state=None)
    assert [r.rule_name for r in results] == ["a"]


def test_duplicate_rule_names_rejected() -> None:
    specs = [RuleSpec(rule=_ConstRule("a", RuleStatus.PASS, 1.0), weight=50.0),
             RuleSpec(rule=_ConstRule("a", RuleStatus.PASS, 1.0), weight=50.0)]
    with pytest.raises(ValueError):
        RuleEngine(specs)


def test_count_by_status() -> None:
    results = [
        RuleResult("a", "technical", RuleStatus.PASS, 1, 1, "r"),
        RuleResult("b", "technical", RuleStatus.PASS, 1, 1, "r"),
        RuleResult("c", "technical", RuleStatus.FAIL, 1, 1, "r"),
        RuleResult("d", "technical", RuleStatus.NOT_APPLICABLE, 0, 0, "r"),
    ]
    passed, failed, na = count_by_status(results)
    assert (passed, failed, na) == (2, 1, 1)


def test_enabled_specs_property() -> None:
    specs = [RuleSpec(rule=_ConstRule("a", RuleStatus.PASS, 1.0), weight=50.0, enabled=True),
             RuleSpec(rule=_ConstRule("b", RuleStatus.PASS, 1.0), weight=50.0, enabled=False)]
    engine = RuleEngine(specs)
    assert [s.rule.name for s in engine.enabled_specs] == ["a"]
