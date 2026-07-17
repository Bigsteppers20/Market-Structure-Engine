"""Tests for swings.py, trend.py, bos.py, choch.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from market_structure import (
    BosEngine,
    ChochEngine,
    EngineConfig,
    SwingDetector,
    TrendDirection,
    TrendEngine,
)
from market_structure.swings import split_swings

from conftest import make_ohlcv


def _zigzag_df(levels: list[float], bars_per_leg: int = 8) -> pd.DataFrame:
    """Build a deterministic zig-zag series moving through given price levels."""
    prices: list[float] = []
    for a, b in zip(levels[:-1], levels[1:]):
        prices.extend(np.linspace(a, b, bars_per_leg, endpoint=False))
    prices.append(levels[-1])
    p = np.array(prices)
    ts = pd.date_range("2025-01-06", periods=len(p), freq="5min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": p,
            "high": p + 0.0002,
            "low": p - 0.0002,
            "close": p,
            "volume": np.full(len(p), 100.0),
        }
    )


def test_swings_detected_on_zigzag() -> None:
    df = _zigzag_df([1.0, 1.01, 1.005, 1.015, 1.008, 1.02])
    swings = SwingDetector(EngineConfig(swing_window=3)).detect(df)
    highs, lows = split_swings(swings)
    assert len(highs) >= 2
    assert len(lows) >= 2
    assert all(s.strength >= 0 for s in swings)
    idx = [s.index for s in swings]
    assert idx == sorted(idx)


def test_swings_empty_on_tiny_input() -> None:
    df = make_ohlcv(5)
    assert SwingDetector(EngineConfig(swing_window=5)).detect(df) == []


def test_swing_distance_from_previous() -> None:
    df = _zigzag_df([1.0, 1.01, 1.0, 1.01, 1.0])
    swings = SwingDetector(EngineConfig(swing_window=3)).detect(df)
    highs, _ = split_swings(swings)
    assert highs[0].distance_from_previous == 0
    for a, b in zip(highs[:-1], highs[1:]):
        assert b.distance_from_previous == b.index - a.index


def test_trend_bullish(trending_up_df: pd.DataFrame) -> None:
    cfg = EngineConfig(swing_window=3)
    swings = SwingDetector(cfg).detect(trending_up_df)
    trend = TrendEngine(cfg).analyze(trending_up_df, swings)
    assert trend.direction == TrendDirection.BULLISH
    assert trend.higher_highs > trend.lower_highs
    assert trend.strength > 0.25
    assert trend.momentum > 0


def test_trend_bearish(trending_down_df: pd.DataFrame) -> None:
    cfg = EngineConfig(swing_window=3)
    swings = SwingDetector(cfg).detect(trending_down_df)
    trend = TrendEngine(cfg).analyze(trending_down_df, swings)
    assert trend.direction == TrendDirection.BEARISH
    assert trend.momentum < 0


def test_trend_sideways_on_flat_range() -> None:
    df = _zigzag_df([1.0, 1.01, 1.0, 1.01, 1.0, 1.01, 1.0])
    cfg = EngineConfig(swing_window=3)
    swings = SwingDetector(cfg).detect(df)
    trend = TrendEngine(cfg).analyze(df, swings)
    assert trend.direction == TrendDirection.SIDEWAYS


def test_bos_bullish_break() -> None:
    # Range then a decisive breakout above the range highs.
    df = _zigzag_df([1.0, 1.01, 1.0, 1.01, 1.0, 1.05])
    cfg = EngineConfig(swing_window=3)
    swings = SwingDetector(cfg).detect(df)
    breaks = BosEngine(cfg).detect(df, swings)
    assert any(b.direction == "bullish" for b in breaks)
    b = [x for x in breaks if x.direction == "bullish"][0]
    assert b.close > b.price
    assert b.strength > 0


def test_bos_bearish_break() -> None:
    df = _zigzag_df([1.05, 1.04, 1.05, 1.04, 1.05, 1.0])
    cfg = EngineConfig(swing_window=3)
    swings = SwingDetector(cfg).detect(df)
    breaks = BosEngine(cfg).detect(df, swings)
    assert any(b.direction == "bearish" for b in breaks)


def test_choch_flags_direction_flip() -> None:
    # Up structure that breaks down at the end -> bearish CHOCH.
    df = _zigzag_df([1.0, 1.02, 1.01, 1.03, 1.02, 1.04, 0.98])
    cfg = EngineConfig(swing_window=3)
    swings = SwingDetector(cfg).detect(df)
    breaks = BosEngine(cfg).detect(df, swings)
    chochs = ChochEngine(cfg).detect(breaks)
    assert chochs, "expected at least one CHOCH"
    assert chochs[-1].direction == "bearish"


def test_choch_empty_without_flip() -> None:
    assert ChochEngine().detect([]) == []
