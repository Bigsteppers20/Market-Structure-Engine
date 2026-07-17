"""Tests for decision_engine.market_analysis -- MARKET ANALYSIS section."""
from __future__ import annotations

from decision_engine.market_analysis import build_market_analysis


class _FakeDirection:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeTrend:
    def __init__(self, direction_name: str = "SIDEWAYS", strength: float = 0.3, valid: bool = True) -> None:
        self.direction = _FakeDirection(direction_name)
        self.strength = strength
        self.valid = valid


class _FakeVolatility:
    def __init__(self, expansion: bool = False, compression: bool = False, valid: bool = True) -> None:
        self.expansion = expansion
        self.compression = compression
        self.valid = valid


class _FakeMarketState:
    def __init__(self, trend=None, volatility=None) -> None:
        self.trend = trend if trend is not None else _FakeTrend()
        self.volatility = volatility if volatility is not None else _FakeVolatility()


def _build(**overrides):
    base = dict(
        market_state=_FakeMarketState(), market_bias="NEUTRAL", consensus_score=50.0,
        regression_available=True, regression_prediction_confidence=70.0,
        regression_confidence_breakdown={"next_close": {"prediction_stability": 80.0}},
        classification_confidence_breakdown={"prediction_stability": 60.0},
    )
    base.update(overrides)
    return build_market_analysis(**base)


def test_agreement_level_buckets() -> None:
    assert _build(consensus_score=90.0).agreement_level == "Strong Agreement"
    assert _build(consensus_score=50.0).agreement_level == "Partial Agreement"
    assert _build(consensus_score=10.0).agreement_level == "Weak Agreement"
    assert _build(consensus_score=0.0).agreement_level == "No Clear Signal"


def test_forecast_quality_buckets() -> None:
    assert _build(regression_prediction_confidence=80.0).forecast_quality == "High"
    assert _build(regression_prediction_confidence=50.0).forecast_quality == "Moderate"
    assert _build(regression_prediction_confidence=10.0).forecast_quality == "Low"


def test_forecast_quality_unavailable_without_regression() -> None:
    result = _build(regression_available=False, regression_prediction_confidence=None)
    assert result.forecast_quality == "Unavailable"


def test_prediction_stability_averages_both_engines() -> None:
    result = _build(
        regression_confidence_breakdown={"next_close": {"prediction_stability": 80.0}},
        classification_confidence_breakdown={"prediction_stability": 60.0},
    )
    assert result.prediction_stability == 70.0


def test_prediction_stability_averages_across_multiple_regression_targets() -> None:
    result = _build(
        regression_confidence_breakdown={
            "next_close": {"prediction_stability": 80.0}, "next_high": {"prediction_stability": 60.0},
        },
        classification_confidence_breakdown=None,
    )
    assert result.prediction_stability == 70.0


def test_prediction_stability_neutral_when_neither_available() -> None:
    result = _build(regression_confidence_breakdown=None, classification_confidence_breakdown=None)
    assert result.prediction_stability == 50.0


def test_market_regime_high_volatility() -> None:
    result = _build(market_state=_FakeMarketState(volatility=_FakeVolatility(expansion=True)))
    assert result.market_regime == "High Volatility"


def test_market_regime_low_volatility() -> None:
    result = _build(market_state=_FakeMarketState(volatility=_FakeVolatility(compression=True)))
    assert result.market_regime == "Low Volatility"


def test_market_regime_trending() -> None:
    result = _build(market_state=_FakeMarketState(trend=_FakeTrend(strength=0.8), volatility=_FakeVolatility()))
    assert result.market_regime == "Trending"


def test_market_regime_ranging_default() -> None:
    result = _build(market_state=_FakeMarketState(trend=_FakeTrend(strength=0.1), volatility=_FakeVolatility()))
    assert result.market_regime == "Ranging"


def test_current_trend_reads_direction_name() -> None:
    result = _build(market_state=_FakeMarketState(trend=_FakeTrend(direction_name="BULLISH")))
    assert result.current_trend == "BULLISH"


def test_market_bias_passthrough() -> None:
    result = _build(market_bias="STRONG_BULLISH")
    assert result.market_bias == "STRONG_BULLISH"


def test_to_dict_serializable() -> None:
    d = _build().to_dict()
    assert set(d) == {"market_bias", "agreement_level", "forecast_quality", "prediction_stability", "market_regime", "current_trend"}
