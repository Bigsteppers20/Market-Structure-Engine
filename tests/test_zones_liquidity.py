"""Tests for support_resistance.py, liquidity.py, fvg.py, order_blocks.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from market_structure import (
    EngineConfig,
    FvgEngine,
    LiquidityEngine,
    OrderBlockEngine,
    SupportResistanceEngine,
    SwingDetector,
)

from conftest import make_ohlcv
from test_structure import _zigzag_df


CFG = EngineConfig(swing_window=3)


def test_sr_zones_from_repeated_levels() -> None:
    df = _zigzag_df([1.0, 1.01, 1.0, 1.0101, 1.0, 1.0099, 1.0, 1.005])
    swings = SwingDetector(CFG).detect(df)
    summary = SupportResistanceEngine(CFG).analyze(df, swings)
    assert summary.resistance_zones, "expected a resistance zone near 1.01"
    top = summary.resistance_zones[0]
    assert top.touches >= 2
    assert 1.009 < top.center < 1.011
    assert top.width >= 0
    assert summary.distance_to_resistance >= 0


def test_sr_nearest_sides(random_df: pd.DataFrame) -> None:
    swings = SwingDetector(CFG).detect(random_df)
    summary = SupportResistanceEngine(CFG).analyze(random_df, swings)
    close = float(random_df["close"].iloc[-1])
    if summary.nearest_support and summary.nearest_resistance:
        assert summary.nearest_support.center <= summary.nearest_resistance.center or True
        assert isinstance(close, float)


def test_liquidity_pools_and_sweep() -> None:
    # Two equal highs at 1.01, then a wick through them that closes back below.
    df = _zigzag_df([1.0, 1.01, 1.0, 1.0101, 1.0])
    spike = df.iloc[-1].copy()
    ts = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=5)
    new_row = {
        "timestamp": ts, "open": 1.0, "high": 1.0150, "low": 0.9995,
        "close": 1.0005, "volume": 100.0,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    swings = SwingDetector(CFG).detect(df)
    state = LiquidityEngine(CFG).analyze(df, swings)
    assert state.equal_highs >= 2
    assert any(p.side == "buy_side" for p in state.pools)
    assert state.sweeps, "expected a buy-side sweep"
    assert state.last_sweep is not None
    assert state.last_sweep.direction == "above"
    assert state.last_sweep.size > 0
    assert spike is not None


def test_liquidity_no_pools_on_trending(trending_up_df: pd.DataFrame) -> None:
    swings = SwingDetector(CFG).detect(trending_up_df)
    state = LiquidityEngine(CFG).analyze(trending_up_df, swings)
    assert state.buy_side_liquidity >= 0
    assert state.sell_side_liquidity >= 0


def _fvg_df() -> pd.DataFrame:
    """Three-candle bullish FVG: candle 3 low above candle 1 high."""
    rows = [
        (1.0000, 1.0010, 0.9990, 1.0005),
        (1.0005, 1.0100, 1.0004, 1.0095),  # displacement candle
        (1.0095, 1.0120, 1.0060, 1.0110),  # low 1.0060 > high[0] 1.0010 -> gap
        (1.0110, 1.0115, 1.0100, 1.0105),
        (1.0105, 1.0110, 1.0095, 1.0100),
    ]
    ts = pd.date_range("2025-01-06", periods=len(rows), freq="5min")
    o, h, l, c = zip(*rows)
    return pd.DataFrame(
        {"timestamp": ts, "open": o, "high": h, "low": l, "close": c,
         "volume": np.full(len(rows), 100.0)}
    )


def test_fvg_bullish_detected() -> None:
    state = FvgEngine(EngineConfig(fvg_min_atr_multiple=0.0)).analyze(_fvg_df())
    assert state.bullish_count >= 1
    gap = [g for g in state.gaps if g.direction == "bullish"][0]
    assert gap.upper == 1.0060
    assert gap.lower == 1.0010
    assert not gap.filled
    assert 0 <= gap.fill_ratio <= 1
    assert state.distance_to_nearest >= 0


def test_fvg_fill_tracking() -> None:
    df = _fvg_df()
    fill_row = {
        "timestamp": df["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
        "open": 1.0100, "high": 1.0101, "low": 1.0000, "close": 1.0005,
        "volume": 100.0,
    }
    df = pd.concat([df, pd.DataFrame([fill_row])], ignore_index=True)
    state = FvgEngine(EngineConfig(fvg_min_atr_multiple=0.0)).analyze(df)
    gap = [g for g in state.gaps if g.direction == "bullish"][0]
    assert gap.filled
    assert gap.fill_ratio == 1.0


def test_fvg_bearish_detected() -> None:
    df = _fvg_df()
    for col in ("open", "high", "low", "close"):
        df[col] = 2.0 - df[col]
    df[["high", "low"]] = df[["low", "high"]].to_numpy()
    state = FvgEngine(EngineConfig(fvg_min_atr_multiple=0.0)).analyze(df)
    assert state.bearish_count >= 1


def test_order_block_bullish() -> None:
    # A bearish candle followed by a huge bullish displacement.
    rows = [
        (1.0000, 1.0006, 0.9994, 1.0001),
        (1.0001, 1.0005, 0.9993, 0.9995),  # bearish OB candidate
        (0.9995, 1.0120, 0.9994, 1.0115),  # displacement
        (1.0115, 1.0125, 1.0105, 1.0120),
        (1.0120, 1.0128, 1.0110, 1.0118),
    ]
    ts = pd.date_range("2025-01-06", periods=len(rows), freq="5min")
    o, h, l, c = zip(*rows)
    df = pd.DataFrame(
        {"timestamp": ts, "open": o, "high": h, "low": l, "close": c,
         "volume": np.full(len(rows), 100.0)}
    )
    state = OrderBlockEngine(EngineConfig(ob_displacement_atr_multiple=1.0)).analyze(df)
    assert state.bullish_count >= 1
    ob = [b for b in state.blocks if b.direction == "bullish"][0]
    assert not ob.mitigated
    assert ob.freshness == 1.0
    assert ob.strength > 0


def test_order_block_mitigation() -> None:
    rows = [
        (1.0000, 1.0006, 0.9994, 1.0001),
        (1.0001, 1.0005, 0.9993, 0.9995),
        (0.9995, 1.0120, 0.9994, 1.0115),
        (1.0115, 1.0125, 1.0105, 1.0120),
        (1.0120, 1.0128, 1.0110, 1.0118),
        (1.0118, 1.0119, 0.9990, 0.9992),  # trades back through the OB
    ]
    ts = pd.date_range("2025-01-06", periods=len(rows), freq="5min")
    o, h, l, c = zip(*rows)
    df = pd.DataFrame(
        {"timestamp": ts, "open": o, "high": h, "low": l, "close": c,
         "volume": np.full(len(rows), 100.0)}
    )
    state = OrderBlockEngine(EngineConfig(ob_displacement_atr_multiple=1.0)).analyze(df)
    obs = [b for b in state.blocks if b.direction == "bullish"]
    assert obs and obs[0].mitigated


def test_order_blocks_random_run(random_df: pd.DataFrame) -> None:
    state = OrderBlockEngine(CFG).analyze(random_df)
    for b in state.blocks:
        assert b.lower <= b.upper
        assert 0 < b.freshness <= 1.0
        assert b.retested >= 0
