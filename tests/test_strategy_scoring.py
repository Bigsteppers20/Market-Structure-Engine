"""Tests for strategy.scoring."""
from __future__ import annotations

import pytest

from strategy.rule_base import RuleResult, RuleStatus
from strategy.scoring import compute_scores, compute_weighted_score


def _r(name, category, status, score, weight):
    return RuleResult(rule_name=name, category=category, status=status, score=score,
                       confidence=80.0, reason="r", weight=weight)


def test_category_scores_isolated_per_category() -> None:
    results = [
        _r("a", "technical", RuleStatus.PASS, 100.0, 50.0),
        _r("b", "market_quality", RuleStatus.PASS, 50.0, 30.0),
        _r("c", "risk", RuleStatus.PASS, 0.0, 20.0),
    ]
    scores = compute_scores(results)
    assert scores.technical_score == pytest.approx(100.0)
    assert scores.market_quality_score == pytest.approx(50.0)
    assert scores.risk_quality_score == pytest.approx(0.0)


def test_overall_score_is_weighted_blend_of_categories() -> None:
    results = [
        _r("a", "technical", RuleStatus.PASS, 100.0, 50.0),
        _r("b", "market_quality", RuleStatus.PASS, 50.0, 30.0),
        _r("c", "risk", RuleStatus.PASS, 0.0, 20.0),
    ]
    scores = compute_scores(results, overall_weights={"technical": 0.5, "market_quality": 0.3, "risk": 0.2})
    assert scores.overall_score == pytest.approx(0.5 * 100.0 + 0.3 * 50.0 + 0.2 * 0.0)


def test_category_score_zero_when_no_applicable_rules_in_category() -> None:
    results = [_r("a", "technical", RuleStatus.NOT_APPLICABLE, 0.0, 50.0)]
    scores = compute_scores(results)
    assert scores.technical_score == 0.0


def test_weighted_score_includes_not_applicable_weight_in_denominator() -> None:
    """Unlike compliance (which renormalizes), weighted_score's denominator
    is the FULL configured weight, so an inapplicable rule still drags it
    down relative to compliance on the same rule set."""
    results = [
        _r("a", "technical", RuleStatus.PASS, 100.0, 50.0),
        _r("b", "technical", RuleStatus.NOT_APPLICABLE, 0.0, 50.0),
    ]
    weighted = compute_weighted_score(results)
    assert weighted == pytest.approx(50.0)  # (50*100 + 50*0) / 100


def test_weighted_score_vs_compliance_differ_with_not_applicable_rules() -> None:
    from strategy.compliance import compute_compliance
    results = [
        _r("a", "technical", RuleStatus.PASS, 100.0, 50.0),
        _r("b", "technical", RuleStatus.NOT_APPLICABLE, 0.0, 50.0),
    ]
    assert compute_compliance(results) == pytest.approx(100.0)
    assert compute_weighted_score(results) == pytest.approx(50.0)


def test_score_breakdown_to_dict() -> None:
    results = [_r("a", "technical", RuleStatus.PASS, 100.0, 100.0)]
    scores = compute_scores(results)
    d = scores.to_dict()
    assert set(d) == {"technical_score", "market_quality_score", "risk_quality_score",
                       "overall_score", "weighted_score"}
