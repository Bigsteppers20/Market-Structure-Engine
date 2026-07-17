"""Tests for candle.py and data_loader.py."""
from __future__ import annotations

import pandas as pd
import pytest

from market_structure import Candle, DataLoader, DataValidationError, EngineConfig
from market_structure.candle import candles_to_dataframe, dataframe_to_candles

from conftest import make_ohlcv


def test_candle_properties() -> None:
    c = Candle(pd.Timestamp("2025-01-01"), open=1.0, high=1.5, low=0.9, close=1.3, volume=10)
    assert c.is_bullish
    assert c.body == pytest.approx(0.3)
    assert c.range == pytest.approx(0.6)
    assert c.upper_wick == pytest.approx(0.2)
    assert c.lower_wick == pytest.approx(0.1)


def test_candle_roundtrip() -> None:
    df = make_ohlcv(20)
    candles = dataframe_to_candles(df)
    back = candles_to_dataframe(candles)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), back[df.columns].reset_index(drop=True)
    )


def test_loader_accepts_candle_list() -> None:
    df = make_ohlcv(30)
    loaded = DataLoader().load(dataframe_to_candles(df))
    assert len(loaded) == 30
    assert list(loaded["timestamp"]) == sorted(loaded["timestamp"])


def test_loader_sorts_and_dedups() -> None:
    df = make_ohlcv(50)
    shuffled = pd.concat([df.iloc[25:], df.iloc[:25], df.iloc[10:12]])
    loaded = DataLoader().load(shuffled)
    assert len(loaded) == 50
    assert loaded["timestamp"].is_monotonic_increasing


def test_loader_missing_columns() -> None:
    with pytest.raises(DataValidationError, match="Missing required"):
        DataLoader().load(pd.DataFrame({"open": [1, 2, 3]}))


def test_loader_rejects_bad_geometry() -> None:
    df = make_ohlcv(10)
    df.loc[5, "high"] = df.loc[5, "low"] - 1.0
    with pytest.raises(DataValidationError, match="high < low"):
        DataLoader().load(df)


def test_loader_rejects_short_input() -> None:
    with pytest.raises(DataValidationError, match="At least 3"):
        DataLoader().load(make_ohlcv(2))


def test_loader_rejects_empty_list() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        DataLoader().load([])


def test_loader_rejects_wrong_type() -> None:
    with pytest.raises(DataValidationError, match="Unsupported"):
        DataLoader().load({"not": "valid"})  # type: ignore[arg-type]


def test_loader_fills_missing_candles() -> None:
    df = make_ohlcv(30)
    gapped = df.drop(index=[10, 11, 20]).reset_index(drop=True)
    cfg = EngineConfig(fill_missing_candles=True)
    loaded = DataLoader(cfg).load(gapped)
    assert len(loaded) == 30
    # Synthetic candle is flat at the prior close with zero volume.
    assert loaded.loc[10, "open"] == loaded.loc[10, "close"]
    assert loaded.loc[10, "volume"] == 0.0


def test_loader_normalizes_string_numbers() -> None:
    df = make_ohlcv(10)
    df["close"] = df["close"].astype(str)
    loaded = DataLoader().load(df)
    assert loaded["close"].dtype == "float64"
