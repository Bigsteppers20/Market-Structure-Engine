"""Tests for strategy.rule_base -- the 19-rule built-in library."""
from __future__ import annotations

import pytest

from strategy.rule_base import (
    BUILTIN_RULES,
    AtrRule,
    BreakOfStructureRule,
    EmaAlignmentRule,
    MacdRule,
    RiskRule,
    RsiRule,
    RuleResult,
    RuleStatus,
    SessionRule,
    SpreadRule,
    TrendRule,
    build_all_rules,
)


def test_builtin_rules_has_exactly_19_entries() -> None:
    expected = {
        "trend", "ema_alignment", "swing_structure", "break_of_structure", "choch",
        "support", "resistance", "liquidity_sweep", "fair_value_gap", "order_block",
        "volume", "atr", "rsi", "macd", "session", "spread", "volatility", "momentum", "risk",
    }
    assert set(BUILTIN_RULES) == expected
    assert len(BUILTIN_RULES) == 19


def test_build_all_rules_returns_every_builtin() -> None:
    rules = build_all_rules()
    assert set(rules) == set(BUILTIN_RULES)
    for name, rule in rules.items():
        assert rule.name == name


def test_build_all_rules_applies_params() -> None:
    rules = build_all_rules({"rsi": {"oversold": 20.0, "overbought": 80.0}})
    assert rules["rsi"].oversold == 20.0
    assert rules["rsi"].overbought == 80.0


def test_every_builtin_rule_returns_valid_result_on_real_market_state(market_state) -> None:
    """Parametrized-by-hand smoke check: every rule in the library must run
    without raising and produce a well-formed RuleResult against a real
    MarketState (not a mock)."""
    rules = build_all_rules()
    for name, rule in rules.items():
        result = rule.evaluate(market_state)
        assert isinstance(result, RuleResult)
        assert result.rule_name == name
        assert result.status in (RuleStatus.PASS, RuleStatus.FAIL, RuleStatus.NOT_APPLICABLE)
        assert 0.0 <= result.score <= 100.0
        assert 0.0 <= result.confidence <= 100.0
        assert result.reason  # non-empty
        if result.status == RuleStatus.NOT_APPLICABLE:
            assert result.score == 0.0 and result.confidence == 0.0


def test_trend_rule_not_applicable_on_insufficient_history() -> None:
    class _FakeTrend:
        valid = False
    class _FakeMarketState:
        trend = _FakeTrend()
    result = TrendRule().evaluate(_FakeMarketState())
    assert result.status == RuleStatus.NOT_APPLICABLE


def test_trend_rule_bullish_direction(bullish_market_state) -> None:
    result = TrendRule().evaluate(bullish_market_state)
    if result.status != RuleStatus.NOT_APPLICABLE:
        # Direction should never be negative on a strongly uptrending series.
        assert result.direction >= 0


def test_ema_alignment_not_applicable_without_warmup() -> None:
    class _FakeState:
        indicators: dict = {}
        indicator_validity: dict = {}
    result = EmaAlignmentRule().evaluate(_FakeState())
    assert result.status == RuleStatus.NOT_APPLICABLE


def test_break_of_structure_not_applicable_when_no_bos() -> None:
    class _FakeStructure:
        last_bos_direction = 0.0
    class _FakeState:
        structure = _FakeStructure()
    result = BreakOfStructureRule().evaluate(_FakeState())
    assert result.status == RuleStatus.NOT_APPLICABLE


def test_rsi_rule_overbought_and_oversold_boundaries() -> None:
    class _FakeState:
        def __init__(self, rsi):
            self.indicators = {"rsi": rsi}
            self.indicator_validity = {"rsi": 1.0}

    over = RsiRule(oversold=30, overbought=70).evaluate(_FakeState(85.0))
    assert over.status == RuleStatus.FAIL
    assert over.direction == -1

    under = RsiRule(oversold=30, overbought=70).evaluate(_FakeState(15.0))
    assert under.status == RuleStatus.FAIL
    assert under.direction == 1

    mid = RsiRule(oversold=30, overbought=70).evaluate(_FakeState(50.0))
    assert mid.status == RuleStatus.PASS


def test_spread_rule_not_applicable_without_spread_data() -> None:
    class _FakeSpread:
        valid = 0.0
    class _FakeState:
        spread = _FakeSpread()
    result = SpreadRule().evaluate(_FakeState())
    assert result.status == RuleStatus.NOT_APPLICABLE


def test_risk_rule_fails_on_spread_spike() -> None:
    class _FakeSpread:
        valid = True
        spike = 1.0
    class _FakeVol:
        valid = True
        expansion = False
    class _FakeSession:
        is_sydney = 0.0; is_asian = 1.0; is_london = 0.0; is_newyork = 0.0
    class _FakeState:
        spread = _FakeSpread()
        volatility = _FakeVol()
        session = _FakeSession()
    result = RiskRule().evaluate(_FakeState())
    assert result.status == RuleStatus.FAIL
    assert "spread spike" in result.reason


def test_risk_rule_passes_with_no_issues() -> None:
    class _FakeSpread:
        valid = True
        spike = 0.0
    class _FakeVol:
        valid = True
        expansion = False
    class _FakeSession:
        is_sydney = 0.0; is_asian = 1.0; is_london = 0.0; is_newyork = 0.0
    class _FakeState:
        spread = _FakeSpread()
        volatility = _FakeVol()
        session = _FakeSession()
    result = RiskRule().evaluate(_FakeState())
    assert result.status == RuleStatus.PASS


def test_rule_result_to_dict_shape() -> None:
    r = RuleResult(rule_name="x", category="technical", status=RuleStatus.PASS,
                    score=80.0, confidence=70.0, reason="ok", weight=10.0, weighted_score=8.0)
    d = r.to_dict()
    assert d["status"] == "PASS"
    assert d["score"] == 80.0
    assert d["metadata"] == {}
