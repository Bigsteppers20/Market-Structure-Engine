"""Tests for strategy.explanation."""
from __future__ import annotations

from strategy.explanation import generate_explanations
from strategy.rule_base import RuleResult, RuleStatus


def _r(name, status, score, weight, reason):
    return RuleResult(rule_name=name, category="technical", status=status, score=score,
                       confidence=80.0, reason=reason, weight=weight)


def test_explanations_include_bias_headline() -> None:
    lines = generate_explanations([], 50.0, 50.0, "BULLISH", [])
    assert lines[0] == "Bullish market structure detected."


def test_passing_rules_sorted_by_weight_descending() -> None:
    results = [
        _r("low", RuleStatus.PASS, 90.0, 10.0, "Low weight passed."),
        _r("high", RuleStatus.PASS, 90.0, 50.0, "High weight passed."),
    ]
    lines = generate_explanations(results, 80.0, 80.0, "BULLISH", [])
    high_idx = lines.index("High weight passed.")
    low_idx = lines.index("Low weight passed.")
    assert high_idx < low_idx


def test_failures_included_as_caveats_capped() -> None:
    results = [_r(f"f{i}", RuleStatus.FAIL, 10.0, 10.0, f"Fail reason {i}.") for i in range(5)]
    lines = generate_explanations(results, 20.0, 20.0, "NEUTRAL", [])
    caveats = [line for line in lines if line.startswith("Caveat:")]
    assert len(caveats) == 3  # MAX_FAILURE_CAVEATS


def test_not_applicable_rules_never_appear() -> None:
    results = [_r("na", RuleStatus.NOT_APPLICABLE, 0.0, 10.0, "Should not show up.")]
    lines = generate_explanations(results, 50.0, 50.0, "NEUTRAL", [])
    assert not any("Should not show up" in line for line in lines)


def test_warnings_included() -> None:
    lines = generate_explanations([], 50.0, 50.0, "NEUTRAL", ["Something is off."])
    assert "Warning: Something is off." in lines


def test_compliance_and_confidence_lines_present() -> None:
    lines = generate_explanations([], 91.7, 88.4, "BULLISH", [])
    assert "Compliance: 92%" in lines
    assert "Confidence: 88%" in lines
