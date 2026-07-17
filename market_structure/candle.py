"""Candle data model.

Defines the :class:`Candle` dataclass, the atomic unit of OHLCV market data
consumed by the Market Structure Engine, plus conversion helpers between
``List[Candle]`` and :class:`pandas.DataFrame`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True, slots=True)
class Candle:
    """A single OHLCV candle.

    Attributes
    ----------
    timestamp:
        Candle open time (timezone-aware or naive ``pd.Timestamp``).
    open, high, low, close:
        Price levels for the candle.
    volume:
        Traded volume (real or tick volume, depending on the feed).
    spread:
        Optional broker spread at candle close, in price units.
    tick_volume:
        Optional tick count for the candle.
    """

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: Optional[float] = field(default=None)
    tick_volume: Optional[float] = field(default=None)

    @property
    def body(self) -> float:
        """Absolute size of the candle body."""
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        """Full high-to-low range of the candle."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """True when the candle closed above its open."""
        return self.close > self.open

    @property
    def upper_wick(self) -> float:
        """Distance from the body top to the high."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Distance from the body bottom to the low."""
        return min(self.open, self.close) - self.low


def candles_to_dataframe(candles: List[Candle]) -> pd.DataFrame:
    """Convert a list of :class:`Candle` objects into an OHLCV DataFrame."""
    records = {
        "timestamp": [c.timestamp for c in candles],
        "open": [c.open for c in candles],
        "high": [c.high for c in candles],
        "low": [c.low for c in candles],
        "close": [c.close for c in candles],
        "volume": [c.volume for c in candles],
    }
    if any(c.spread is not None for c in candles):
        records["spread"] = [c.spread for c in candles]
    if any(c.tick_volume is not None for c in candles):
        records["tick_volume"] = [c.tick_volume for c in candles]
    return pd.DataFrame(records)


def dataframe_to_candles(df: pd.DataFrame) -> List[Candle]:
    """Convert a validated OHLCV DataFrame into a list of :class:`Candle`."""
    has_spread = "spread" in df.columns
    has_tick = "tick_volume" in df.columns
    out: List[Candle] = []
    for row in df.itertuples(index=False):
        out.append(
            Candle(
                timestamp=pd.Timestamp(row.timestamp),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                spread=float(row.spread) if has_spread else None,
                tick_volume=float(row.tick_volume) if has_tick else None,
            )
        )
    return out
