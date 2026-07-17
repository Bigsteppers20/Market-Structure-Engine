"""Integration tests for engine.py + feature_vector.py, plus performance."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from market_structure import (
    EngineConfig,
    EngineStateError,
    MarketStructureEngine,
)
from market_structure.candle import dataframe_to_candles
from market_structure.feature_vector import build_session

from conftest import make_ohlcv


def test_public_api_flow(random_df: pd.DataFrame) -> None:
    engine = MarketStructureEngine()
    engine.load(random_df)
    state = engine.analyze()
    assert state is engine.market_state()
    vec, names = engine.feature_vector()
    assert len(vec) == len(names)
    assert len(names) == len(set(names)), "duplicate feature names"
    assert vec.dtype == np.float64
    assert np.isfinite(vec).all(), "feature vector must contain only finite values"


def test_engine_accepts_candle_list(random_df: pd.DataFrame) -> None:
    engine = MarketStructureEngine()
    engine.load(dataframe_to_candles(random_df))
    state = engine.analyze()
    assert state.n_candles == len(random_df)


def test_engine_state_errors() -> None:
    engine = MarketStructureEngine()
    with pytest.raises(EngineStateError):
        engine.analyze()
    with pytest.raises(EngineStateError):
        engine.market_state()
    with pytest.raises(EngineStateError):
        _ = engine.data


def test_no_trade_signals_in_features(random_df: pd.DataFrame) -> None:
    """Hard requirement: the output must never contain trade decisions."""
    engine = MarketStructureEngine().load(random_df)
    engine.analyze()
    _, names = engine.feature_vector()
    # Descriptive terms like "buy_side liquidity" or "macd_signal" are fine;
    # decision-style names are not.
    forbidden = {
        "buy", "sell", "no_trade", "trade_signal", "buy_signal", "sell_signal",
        "prediction", "action", "decision", "entry", "exit",
    }
    for name in names:
        tokens = set(name.lower().split("_"))
        assert name.lower() not in forbidden, name
        assert not ({"prediction", "decision", "entry"} & tokens), name
        assert "no_trade" not in name.lower(), name


def test_feature_vector_deterministic(random_df: pd.DataFrame) -> None:
    e1 = MarketStructureEngine().load(random_df)
    e1.analyze()
    v1, n1 = e1.feature_vector()
    e2 = MarketStructureEngine().load(random_df)
    e2.analyze()
    v2, n2 = e2.feature_vector()
    assert n1 == n2
    np.testing.assert_allclose(v1, v2)


def test_trend_and_structure_populated(trending_up_df: pd.DataFrame) -> None:
    engine = MarketStructureEngine(EngineConfig(swing_window=3)).load(trending_up_df)
    state = engine.analyze()
    assert state.trend is not None and int(state.trend.direction) == 1
    d = state.to_dict()
    assert d["trend_direction"] == 1.0
    assert d["structure_bullish_bos_count"] >= 1.0
    assert d["pa_current_close"] == pytest.approx(float(trending_up_df["close"].iloc[-1]))


def test_session_features() -> None:
    df = make_ohlcv(10)
    df["timestamp"] = pd.date_range("2025-01-06 13:30", periods=10, freq="5min")
    feats = build_session(df, EngineConfig())
    assert feats.is_london == 1.0
    assert feats.is_newyork == 1.0
    # session_overlap was removed (Task 2: exact function of the 4 is_*
    # flags, >= 2 active) -- verify the underlying condition directly.
    assert (feats.is_sydney + feats.is_asian + feats.is_london + feats.is_newyork) >= 2.0
    assert feats.hour == 14.0
    assert feats.day_of_week == 0.0  # Monday


def test_session_wraps_midnight() -> None:
    df = make_ohlcv(10)
    df["timestamp"] = pd.date_range("2025-01-06 22:00", periods=10, freq="5min")
    feats = build_session(df, EngineConfig())
    assert feats.is_sydney == 1.0
    assert feats.is_london == 0.0


def test_microstructure_populated(random_df: pd.DataFrame) -> None:
    engine = MarketStructureEngine(EngineConfig(swing_window=3)).load(random_df)
    state = engine.analyze()
    m = state.microstructure
    assert m.average_swing_length > 0
    assert m.time_between_swings > 0
    assert 0 <= m.retracement_pct <= 100


def test_reload_resets_state(random_df: pd.DataFrame, trending_up_df: pd.DataFrame) -> None:
    engine = MarketStructureEngine()
    engine.load(random_df)
    engine.analyze()
    engine.load(trending_up_df)
    with pytest.raises(EngineStateError):
        engine.market_state()


@pytest.mark.performance
def test_performance_100k_under_2s() -> None:
    df = make_ohlcv(100_000, seed=11)
    engine = MarketStructureEngine()
    engine.load(df)
    start = time.perf_counter()
    engine.analyze()
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"analyze() took {elapsed:.2f}s for 100k candles"
