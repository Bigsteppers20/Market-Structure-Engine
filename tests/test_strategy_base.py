"""Tests for strategy.strategy_base: bias/recommendation logic and the full
TradingStrategy.evaluate() contract, exercised via a minimal concrete
strategy built directly on the built-in rule library."""
from __future__ import annotations

from typing import Dict

import pytest

from strategy.config import StrategyConfig
from strategy.rule_base import Rule, RuleResult, RuleStatus, build_all_rules
from strategy.strategy_base import (
    MarketBias,
    TradeRecommendation,
    TradingStrategy,
    compute_market_bias,
    compute_recommendation,
)


def _r(name, status, score, weight, direction=0):
    return RuleResult(rule_name=name, category="technical", status=status, score=score,
                       confidence=80.0, reason="r", weight=weight, metadata={"direction": direction})


# --------------------------------------------------------------------------- #
# compute_market_bias
# --------------------------------------------------------------------------- #
def test_bias_strong_bullish_when_all_directional_rules_agree_strongly() -> None:
    results = [_r("a", RuleStatus.PASS, 100.0, 100.0, direction=1)]
    assert compute_market_bias(results) == MarketBias.STRONG_BULLISH


def test_bias_strong_bearish() -> None:
    results = [_r("a", RuleStatus.PASS, 100.0, 100.0, direction=-1)]
    assert compute_market_bias(results) == MarketBias.STRONG_BEARISH


def test_bias_neutral_with_no_directional_rules() -> None:
    results = [_r("a", RuleStatus.PASS, 100.0, 100.0, direction=0)]
    assert compute_market_bias(results) == MarketBias.NEUTRAL


def test_bias_neutral_when_directions_cancel_out() -> None:
    results = [_r("a", RuleStatus.PASS, 100.0, 50.0, direction=1),
               _r("b", RuleStatus.PASS, 100.0, 50.0, direction=-1)]
    assert compute_market_bias(results) == MarketBias.NEUTRAL


def test_bias_mild_bullish() -> None:
    # net ~ 0.3 -> BULLISH band, not STRONG_BULLISH
    results = [_r("a", RuleStatus.PASS, 30.0, 100.0, direction=1)]
    assert compute_market_bias(results) == MarketBias.BULLISH


def test_bias_ignores_not_applicable() -> None:
    results = [_r("a", RuleStatus.NOT_APPLICABLE, 100.0, 100.0, direction=1)]
    assert compute_market_bias(results) == MarketBias.NEUTRAL


# --------------------------------------------------------------------------- #
# compute_recommendation
# --------------------------------------------------------------------------- #
def test_recommendation_no_trade_on_neutral_bias() -> None:
    cfg = StrategyConfig(strategy_name="t", rule_weights={"trend": 100.0})
    rec = compute_recommendation(MarketBias.NEUTRAL, 90.0, 90.0, cfg, [])
    assert rec == TradeRecommendation.NO_TRADE


def test_recommendation_no_trade_when_risk_rule_fails() -> None:
    cfg = StrategyConfig(strategy_name="t", rule_weights={"trend": 100.0})
    risk_fail = [_r("risk", RuleStatus.FAIL, 10.0, 10.0)]
    rec = compute_recommendation(MarketBias.STRONG_BULLISH, 95.0, 95.0, cfg, risk_fail)
    assert rec == TradeRecommendation.NO_TRADE


def test_recommendation_buy_when_bullish_and_thresholds_met() -> None:
    cfg = StrategyConfig(strategy_name="t", rule_weights={"trend": 100.0},
                          compliance_threshold=70.0, confidence_threshold=60.0)
    rec = compute_recommendation(MarketBias.BULLISH, 80.0, 70.0, cfg, [])
    assert rec == TradeRecommendation.BUY


def test_recommendation_sell_when_bearish_and_thresholds_met() -> None:
    cfg = StrategyConfig(strategy_name="t", rule_weights={"trend": 100.0},
                          compliance_threshold=70.0, confidence_threshold=60.0)
    rec = compute_recommendation(MarketBias.BEARISH, 80.0, 70.0, cfg, [])
    assert rec == TradeRecommendation.SELL


def test_recommendation_wait_when_thresholds_not_met() -> None:
    cfg = StrategyConfig(strategy_name="t", rule_weights={"trend": 100.0},
                          compliance_threshold=90.0, confidence_threshold=90.0)
    rec = compute_recommendation(MarketBias.BULLISH, 50.0, 50.0, cfg, [])
    assert rec == TradeRecommendation.WAIT


# --------------------------------------------------------------------------- #
# Full TradingStrategy.evaluate() integration, via a minimal concrete strategy
# --------------------------------------------------------------------------- #
class _MiniStrategy(TradingStrategy):
    def build_rules(self) -> Dict[str, Rule]:
        return build_all_rules(self.config.rule_params)


def _mini_config(**overrides) -> StrategyConfig:
    base = dict(
        strategy_name="mini", rule_weights={"trend": 40.0, "rsi": 30.0, "session": 30.0},
        compliance_threshold=50.0, confidence_threshold=50.0,
    )
    base.update(overrides)
    return StrategyConfig(**base)


def test_evaluate_returns_complete_evaluation(market_state) -> None:
    strategy = _MiniStrategy(_mini_config())
    evaluation = strategy.evaluate(market_state, symbol="EUR_USD", timeframe="M5")

    assert evaluation.strategy_name == "mini"
    assert evaluation.symbol == "EUR_USD"
    assert evaluation.timeframe == "M5"
    assert evaluation.market_bias in {b.value for b in MarketBias}
    assert evaluation.recommendation in {r.value for r in TradeRecommendation}
    assert 0.0 <= evaluation.strategy_compliance <= 100.0
    assert 0.0 <= evaluation.strategy_confidence <= 100.0
    assert evaluation.rules_passed + evaluation.rules_failed + evaluation.rules_not_applicable == 3
    assert len(evaluation.rule_results) == 3
    assert evaluation.explanations
    assert evaluation.explanations[-3].startswith("Compliance:")
    assert evaluation.explanations[-2].startswith("Confidence:")
    assert evaluation.explanations[-1].startswith("MSE Compliance:")
    assert 0.0 <= evaluation.mse_compliance <= 100.0


def test_evaluate_to_dict_matches_spec_shape(market_state) -> None:
    strategy = _MiniStrategy(_mini_config())
    d = strategy.evaluate(market_state).to_dict()
    required_keys = {
        "strategy_name", "strategy_version", "timestamp", "symbol", "timeframe",
        "market_bias", "recommendation", "strategy_compliance", "strategy_confidence",
        "technical_score", "market_quality_score", "risk_quality_score", "overall_score",
        "rules_passed", "rules_failed", "rules_not_applicable", "warnings",
        "rule_results", "explanations", "weighted_score",
    }
    assert required_keys <= set(d)


def test_rule_weight_not_in_build_rules_raises() -> None:
    with pytest.raises(Exception):
        _MiniStrategy(StrategyConfig(strategy_name="bad", rule_weights={"not_a_real_rule": 100.0}))


def test_disabling_a_rule_excludes_it_from_results(market_state) -> None:
    cfg = _mini_config()
    cfg.enabled_rules["session"] = False
    strategy = _MiniStrategy(cfg)
    evaluation = strategy.evaluate(market_state)
    assert "session" not in [r.rule_name for r in evaluation.rule_results]
    assert len(evaluation.rule_results) == 2


def test_low_candle_count_produces_warning() -> None:
    from conftest import make_ohlcv
    from market_structure import EngineConfig, MarketStructureEngine

    df = make_ohlcv(30, seed=1)
    engine = MarketStructureEngine(EngineConfig(swing_window=3))
    engine.load(df)
    engine.analyze()
    small_state = engine.market_state()

    strategy = _MiniStrategy(_mini_config())
    evaluation = strategy.evaluate(small_state)
    assert any("candles loaded" in w for w in evaluation.warnings)


def test_evaluate_rejects_non_market_state() -> None:
    strategy = _MiniStrategy(_mini_config())
    with pytest.raises(Exception):
        strategy.evaluate("not a market state")
