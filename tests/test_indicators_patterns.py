"""Tests for indicators.py and candle_patterns.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from market_structure import CandlePatternEngine, EngineConfig, IndicatorEngine
from market_structure.indicators import ema, rsi, sma

from conftest import make_ohlcv


def test_sma_matches_pandas() -> None:
    x = np.arange(1.0, 51.0)
    expected = pd.Series(x).rolling(10, min_periods=1).mean().to_numpy()
    np.testing.assert_allclose(sma(x, 10), expected)


def test_ema_converges_on_constant() -> None:
    x = np.full(200, 5.0)
    np.testing.assert_allclose(ema(x, 20), 5.0)


def test_rsi_bounds_and_direction() -> None:
    up = np.linspace(1.0, 2.0, 100)
    down = np.linspace(2.0, 1.0, 100)
    rsi_up = rsi(up, 14)
    rsi_down = rsi(down, 14)
    assert np.nanmax(rsi_up) <= 100.0 and np.nanmin(rsi_up) >= 0.0
    assert np.nanmean(rsi_up[20:]) > 70
    assert np.nanmean(rsi_down[20:]) < 30


def test_indicator_panel_complete(random_df: pd.DataFrame) -> None:
    panel = IndicatorEngine(EngineConfig()).analyze(random_df)
    required = {
        "ema_20", "ema_50", "ema_100", "ema_200", "sma", "atr", "rsi",
        "macd", "macd_signal", "adx", "momentum", "roc",
        "cci", "stoch_k", "stoch_d", "williams_r", "bb_upper",
        "bb_lower", "std_dev", "volatility", "rolling_median",
        "vwap", "true_range",
        "volume_ma", "relative_volume", "volume_spike", "volume_trend",
        "volume_delta", "volume_ratio", "tick_volume",
    }
    # macd_histogram, rolling_mean, rolling_variance and bb_middle were
    # removed as exact duplicates/linear combinations of features already
    # present (see FEATURE_OPTIMIZATION_REPORT.md, Tasks 1-2).
    assert required <= set(panel.series)
    assert not ({"macd_histogram", "rolling_mean", "rolling_variance", "bb_middle"} & set(panel.series))
    n = len(random_df)
    for name, arr in panel.series.items():
        assert len(arr) == n, name
        assert np.isfinite(panel.snapshot[name]), name
    # Every indicator carries a validity flag (Task 3: no silent 0.0 for
    # "not enough history yet").
    assert required <= set(panel.valid)
    for name in required:
        assert panel.valid[name] in (0.0, 1.0), name


def test_indicator_relationships(random_df: pd.DataFrame) -> None:
    panel = IndicatorEngine(EngineConfig()).analyze(random_df)
    s = panel.series
    # bb_middle is no longer stored, but bb_upper/bb_lower must still bracket
    # the (removed) middle band symmetrically around it.
    mid = (s["bb_upper"] + s["bb_lower"]) / 2.0
    np.testing.assert_allclose(mid, s["sma"], atol=1e-9)
    assert (s["bb_upper"] >= mid).all()
    assert (mid >= s["bb_lower"]).all()
    valid = np.isfinite(s["stoch_k"])
    assert (s["stoch_k"][valid] >= 0).all() and (s["stoch_k"][valid] <= 100).all()
    wr = s["williams_r"][np.isfinite(s["williams_r"])]
    assert (wr <= 0).all() and (wr >= -100).all()


def test_indicator_validity_warms_up() -> None:
    """A short series should mark long-period indicators invalid, not 0.0-as-real."""
    df = make_ohlcv(30)
    panel = IndicatorEngine(EngineConfig()).analyze(df)
    assert panel.valid["ema_200"] == 0.0
    assert panel.valid["ema_20"] == 1.0
    assert panel.valid["tick_volume"] == 0.0  # no tick_volume column supplied


def test_volume_spike_flag() -> None:
    df = make_ohlcv(60)
    df.loc[len(df) - 1, "volume"] = df["volume"].mean() * 10
    panel = IndicatorEngine(EngineConfig()).analyze(df)
    assert panel.snapshot["volume_spike"] == 1.0
    assert panel.snapshot["relative_volume"] > 2.0


def _pattern_df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    ts = pd.date_range("2025-01-06", periods=len(rows), freq="5min")
    o, h, l, c = zip(*rows)
    return pd.DataFrame(
        {"timestamp": ts, "open": o, "high": h, "low": l, "close": c,
         "volume": np.full(len(rows), 100.0)}
    )


def test_hammer_and_doji() -> None:
    df = _pattern_df([
        (1.000, 1.001, 0.999, 1.0005),
        (1.0005, 1.0006, 0.990, 1.0004),  # long lower wick, tiny body: hammer+doji family
    ])
    panel = CandlePatternEngine().analyze(df)
    assert panel.snapshot["hammer"] == 1.0
    assert panel.snapshot["bullish_pin_bar"] == 1.0


def test_engulfing_patterns() -> None:
    df = _pattern_df([
        (1.0010, 1.0012, 0.9998, 1.0000),  # bearish
        (0.9999, 1.0025, 0.9998, 1.0020),  # bullish engulfs prior body
    ])
    panel = CandlePatternEngine().analyze(df)
    assert panel.snapshot["bullish_engulfing"] == 1.0
    # No composite bullish_score anymore (Task 2: redundant with the
    # individual flags it summed) -- verify the flag directly instead.


def test_inside_outside_bar() -> None:
    df = _pattern_df([
        (1.000, 1.010, 0.990, 1.005),
        (1.004, 1.006, 1.002, 1.005),  # inside
        (1.005, 1.020, 0.985, 1.015),  # outside
    ])
    panel = CandlePatternEngine().analyze(df)
    assert panel.series["inside_bar"][1] == 1.0
    assert panel.series["outside_bar"][2] == 1.0


def test_marubozu_and_soldiers() -> None:
    df = _pattern_df([
        (1.000, 1.0100, 0.9999, 1.0100),
        (1.0100, 1.0200, 1.0099, 1.0200),
        (1.0200, 1.0300, 1.0199, 1.0300),
    ])
    panel = CandlePatternEngine().analyze(df)
    assert panel.snapshot["bullish_marubozu"] == 1.0
    assert panel.snapshot["three_white_soldiers"] == 1.0


def test_all_patterns_present(random_df: pd.DataFrame) -> None:
    panel = CandlePatternEngine().analyze(random_df)
    expected = {
        "hammer", "inverted_hammer", "shooting_star", "doji", "dragonfly_doji",
        "gravestone_doji", "bullish_engulfing", "bearish_engulfing",
        "bullish_harami", "bearish_harami",
        "morning_star", "evening_star", "piercing_line",
        "dark_cloud_cover", "three_white_soldiers", "three_black_crows",
        "inside_bar", "outside_bar", "bullish_pin_bar", "bearish_pin_bar",
        "bullish_marubozu", "bearish_marubozu", "spinning_top",
    }
    # harami/pin_bar/marubozu unions were removed as exact duplicates of
    # (bullish_X | bearish_X), which already appear individually above
    # (see FEATURE_OPTIMIZATION_REPORT.md, Task 2).
    assert expected <= set(panel.series)
    assert not ({"harami", "pin_bar", "marubozu"} & set(panel.series))
    for name, arr in panel.series.items():
        assert set(np.unique(arr)) <= {0.0, 1.0}, name
