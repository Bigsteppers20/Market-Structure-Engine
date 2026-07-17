"""Break of Structure (BOS) detection.

A bullish BOS occurs when a candle *closes* above the most recent confirmed
swing high; a bearish BOS when a candle closes below the most recent
confirmed swing low. Strength is the ATR-normalized breach distance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import pandas as pd

from .config import EngineConfig
from .swings import SwingPoint, split_swings
from .utils import true_range, wilder_smooth

Direction = Literal["bullish", "bearish"]


@dataclass(slots=True)
class StructureBreak:
    """A confirmed break of structure.

    Attributes
    ----------
    direction:
        ``"bullish"`` (close above prior swing high) or ``"bearish"``.
    index:
        Bar index of the breaking close.
    price:
        The structural level that was broken.
    close:
        Closing price of the breaking candle.
    strength:
        ATR-normalized distance between the breaking close and the level.
    timestamp:
        Timestamp of the breaking candle.
    swing_index:
        Bar index of the swing whose level was broken.
    """

    direction: Direction
    index: int
    price: float
    close: float
    strength: float
    timestamp: pd.Timestamp
    swing_index: int


class BosEngine:
    """Detects every bullish/bearish BOS across the series."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def detect(self, df: pd.DataFrame, swings: List[SwingPoint]) -> List[StructureBreak]:
        """Scan the series once, tracking the active swing high/low levels.

        The scan advances swing-by-swing (swing counts are small), and inside
        each inter-swing segment uses vectorized ``argmax`` on the closes.
        """
        highs, lows = split_swings(swings)
        if not highs and not lows:
            return []
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        atr = wilder_smooth(true_range(high, low, close), self.config.atr_period)
        atr = np.where(np.isfinite(atr) & (atr > 0), atr, np.nanmedian(high - low) or 1.0)
        timestamps = df["timestamp"].to_numpy()
        k = self.config.swing_window

        events: List[StructureBreak] = []
        hi_ptr = lo_ptr = 0
        active_high: SwingPoint | None = None
        active_low: SwingPoint | None = None
        n = len(df)
        i = 0
        while i < n:
            # Activate swings once they are confirmed (k bars after pivot).
            while hi_ptr < len(highs) and highs[hi_ptr].index + k <= i:
                active_high = highs[hi_ptr]
                hi_ptr += 1
            while lo_ptr < len(lows) and lows[lo_ptr].index + k <= i:
                active_low = lows[lo_ptr]
                lo_ptr += 1

            next_conf = min(
                highs[hi_ptr].index + k if hi_ptr < len(highs) else n,
                lows[lo_ptr].index + k if lo_ptr < len(lows) else n,
                n,
            )
            seg = slice(i, max(next_conf, i + 1))
            seg_close = close[seg]

            if active_high is not None:
                brk = np.nonzero(seg_close > active_high.price)[0]
                if brk.size:
                    j = i + int(brk[0])
                    events.append(
                        StructureBreak(
                            direction="bullish",
                            index=j,
                            price=active_high.price,
                            close=float(close[j]),
                            strength=float((close[j] - active_high.price) / atr[j]),
                            timestamp=pd.Timestamp(timestamps[j]),
                            swing_index=active_high.index,
                        )
                    )
                    active_high = None
            if active_low is not None:
                brk = np.nonzero(seg_close < active_low.price)[0]
                if brk.size:
                    j = i + int(brk[0])
                    events.append(
                        StructureBreak(
                            direction="bearish",
                            index=j,
                            price=active_low.price,
                            close=float(close[j]),
                            strength=float((active_low.price - close[j]) / atr[j]),
                            timestamp=pd.Timestamp(timestamps[j]),
                            swing_index=active_low.index,
                        )
                    )
                    active_low = None
            i = max(next_conf, i + 1)

        events.sort(key=lambda e: e.index)
        return events
