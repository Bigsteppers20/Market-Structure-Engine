"""Tests for strategy.compliance and strategy.confidence."""
from __future__ import annotations

import pytest

from strategy.compliance import compute_compliance
from strategy.confidence import compute_confidence
from strategy.rule_base import RuleResult, RuleStatus


def _r(name, status, score, confidence, weight, direction=0):
    return RuleResult(rule_name=name, category="technical", status=status, score=score,
                       confidence=confidence, reason="r", weight=weight,
                       metadata={"direction": direction})


def test_compliance_is_weighted_average_of_applicable() -> None:
    results = [_r("a", RuleStatus.PASS, 100.0, 90.0, 60.0), _r("b", RuleStatus.FAIL, 0.0, 50.0, 40.0)]
    compliance = compute_compliance(results)
    assert compliance == pytest.approx(60.0)  # (60*100 + 40*0) / 100


def test_compliance_excludes_not_applicable_from_denominator() -> None:
    results = [_r("a", RuleStatus.PASS, 100.0, 90.0, 50.0),
               _r("b", RuleStatus.NOT_APPLICABLE, 0.0, 0.0, 50.0)]
    compliance = compute_compliance(results)
    assert compliance == pytest.approx(100.0)  # only "a" counts, weight renormalized


def test_compliance_all_not_applicable_is_zero() -> None:
    results = [_r("a", RuleStatus.NOT_APPLICABLE, 0.0, 0.0, 100.0)]
    assert compute_compliance(results) == 0.0


def test_compliance_empty_is_zero() -> None:
    assert compute_compliance([]) == 0.0


def test_confidence_is_not_identical_to_compliance(market_state) -> None:
    """The two must be computed from genuinely different inputs."""
    results = [_r("a", RuleStatus.PASS, 100.0, 40.0, 100.0, direction=1)]
    compliance = compute_compliance(results)
    confidence = compute_confidence(results, market_state).overall
    assert compliance == pytest.approx(100.0)
    assert confidence != pytest.approx(compliance)  # low rule confidence pulls this down


def test_confidence_breakdown_has_six_components(market_state) -> None:
    results = [_r("a", RuleStatus.PASS, 80.0, 70.0, 100.0, direction=1)]
    breakdown = compute_confidence(results, market_state)
    d = breakdown.to_dict()
    assert set(d) == {
        "rule_certainty", "indicator_agreement", "trend_consistency",
        "volatility_stability", "market_structure_quality", "signal_stability",
    }
    assert 0.0 <= breakdown.overall <= 100.0


def test_indicator_agreement_full_when_all_directions_agree(market_state) -> None:
    results = [_r("a", RuleStatus.PASS, 80.0, 70.0, 50.0, direction=1),
               _r("b", RuleStatus.PASS, 80.0, 70.0, 50.0, direction=1)]
    breakdown = compute_confidence(results, market_state)
    assert breakdown.indicator_agreement == pytest.approx(100.0)


def test_indicator_agreement_split_when_directions_disagree(market_state) -> None:
    results = [_r("a", RuleStatus.PASS, 80.0, 70.0, 50.0, direction=1),
               _r("b", RuleStatus.PASS, 80.0, 70.0, 50.0, direction=-1)]
    breakdown = compute_confidence(results, market_state)
    assert breakdown.indicator_agreement == pytest.approx(50.0)


def test_confidence_no_directional_rules_is_neutral(market_state) -> None:
    results = [_r("a", RuleStatus.PASS, 80.0, 70.0, 100.0, direction=0)]
    breakdown = compute_confidence(results, market_state)
    assert breakdown.indicator_agreement == pytest.approx(50.0)
