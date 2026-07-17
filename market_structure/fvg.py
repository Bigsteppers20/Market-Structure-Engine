"""Fair Value Gap (FVG) detection.

A bullish FVG exists when ``low[i] > high[i-2]`` (a gap the middle candle's
impulse left unfilled); bearish when ``high[i] < low[i-2]``. Detection and
fill-tracking are fully vectorized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from .config import EngineConfig
from .utils import last_valid, rolling_max, rolling_min, true_range, wilder_smooth

Direction = Literal["bullish", "bearish"]


@dataclass(slots=True)
class FairValueGap:
    """A three-candle fair value gap.

    Attributes
    ----------
    direction:
        ``"bullish"`` or ``"bearish"``.
    index:
        Bar index of the third candle (gap confirmation bar).
    upper, lower:
        Price bounds of the gap.
    size:
        Gap height in price units (``upper - lower``).
    size_atr:
        Gap height normalized by ATR at formation.
    age:
        Bars elapsed since formation (relative to series end).
    filled:
        True once price has fully traded through the gap.
    fill_ratio:
        Fraction of the gap that has been filled, in [0, 1].
    timestamp:
        Timestamp of the confirmation bar.
    """

    direction: Direction
    index: int
    upper: float
    lower: float
    size: float
    size_atr: float
    age: int
    filled: bool
    fill_ratio: float
    timestamp: pd.Timestamp


@dataclass(slots=True)
class FvgState:
    """All gaps plus proximity of the nearest unfilled gap."""

    gaps: List[FairValueGap]
    bullish_count: int
    bearish_count: int
    unfilled_count: int
    nearest: Optional[FairValueGap]
    distance_to_nearest: float
    """ATR-normalized distance from close to the nearest unfilled gap edge."""


class FvgEngine:
    """Detects FVGs and tracks their fill state."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def analyze(self, df: pd.DataFrame) -> FvgState:
        """Vectorized FVG scan across the whole series."""
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        n = len(df)
        atr = wilder_smooth(true_range(high, low, close), self.config.atr_period)
        atr_fallback = float(np.median(high - low)) or 1.0
        atr = np.where(np.isfinite(atr) & (atr > 0), atr, atr_fallback)
        timestamps = df["timestamp"].to_numpy()

        bull = np.zeros(n, dtype=bool)
        bear = np.zeros(n, dtype=bool)
        bull[2:] = low[2:] > high[:-2]
        bear[2:] = high[2:] < low[:-2]

        min_size = atr * self.config.fvg_min_atr_multiple
        bull_size = np.zeros(n)
        bear_size = np.zeros(n)
        bull_size[2:] = low[2:] - high[:-2]
        bear_size[2:] = low[:-2] - high[2:]
        bull &= bull_size >= min_size
        bear &= bear_size >= min_size

        # Suffix extremes let us test "was the gap traded through later?" in O(1).
        suffix_min_low = rolling_min(low[::-1], n)[::-1]   # min(low[i:])
        suffix_max_high = rolling_max(high[::-1], n)[::-1]  # max(high[i:])

        gaps: List[FairValueGap] = []
        for direction, mask, sizes in (("bullish", bull, bull_size), ("bearish", bear, bear_size)):
            for i in np.nonzero(mask)[0]:
                i = int(i)
                if direction == "bullish":
                    upper, lower = float(low[i]), float(high[i - 2])
                    future_min = suffix_min_low[i + 1] if i + 1 < n else np.inf
                    filled = bool(future_min <= lower)
                    fill_ratio = float(np.clip((upper - future_min) /
                                       (upper - lower), 0, 1)) if i + 1 < n else 0.0
                else:
                    upper, lower = float(low[i - 2]), float(high[i])
                    future_max = suffix_max_high[i + 1] if i + 1 < n else -np.inf
                    filled = bool(future_max >= upper)
                    fill_ratio = float(np.clip((future_max - lower) /
                                       (upper - lower), 0, 1)) if i + 1 < n else 0.0
                gaps.append(
                    FairValueGap(
                        direction=direction,  # type: ignore[arg-type]
                        index=i,
                        upper=upper,
                        lower=lower,
                        size=float(sizes[i]),
                        size_atr=float(sizes[i] / atr[i]),
                        age=n - 1 - i,
                        filled=filled,
                        fill_ratio=fill_ratio,
                        timestamp=pd.Timestamp(timestamps[i]),
                    )
                )
        gaps.sort(key=lambda g: g.index)

        last_close = float(close[-1])
        atr_now = last_valid(atr, atr_fallback)
        unfilled = [g for g in gaps if not g.filled]
        nearest = min(
            unfilled,
            key=lambda g: min(abs(last_close - g.upper), abs(last_close - g.lower)),
            default=None,
        )
        dist = (
            min(abs(last_close - nearest.upper), abs(last_close - nearest.lower)) / atr_now
            if nearest
            else 0.0
        )
        return FvgState(
            gaps=gaps,
            bullish_count=sum(g.direction == "bullish" for g in gaps),
            bearish_count=sum(g.direction == "bearish" for g in gaps),
            unfilled_count=len(unfilled),
            nearest=nearest,
            distance_to_nearest=float(dist),
        )
