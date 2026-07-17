"""Rule-based Order Block detection.

A bullish order block is the last bearish candle before a strong bullish
displacement (impulse > ``ob_displacement_atr_multiple`` ATRs within the
following candles); a bearish order block is the mirror image. Tracks
mitigation (price returning into the zone) and retests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from .config import EngineConfig
from .utils import last_valid, true_range, wilder_smooth

Direction = Literal["bullish", "bearish"]


@dataclass(slots=True)
class OrderBlock:
    """A detected order block zone.

    Attributes
    ----------
    direction:
        ``"bullish"`` (demand) or ``"bearish"`` (supply).
    index:
        Bar index of the order block candle.
    upper, lower:
        Zone bounds (candle high/low of the OB candle).
    strength:
        ATR-normalized size of the displacement that validated the block.
    age:
        Bars since formation (relative to series end).
    freshness:
        1.0 for untouched blocks, decaying toward 0 with each retest.
    mitigated:
        True once price has traded back through the zone's origin side.
    retested:
        Number of times price re-entered the zone after formation.
    distance:
        ATR-normalized distance from the last close to the zone edge.
    timestamp:
        Timestamp of the order block candle.
    """

    direction: Direction
    index: int
    upper: float
    lower: float
    strength: float
    age: int
    freshness: float
    mitigated: bool
    retested: int
    distance: float
    timestamp: pd.Timestamp


@dataclass(slots=True)
class OrderBlockState:
    """All order blocks plus the nearest unmitigated block."""

    blocks: List[OrderBlock]
    bullish_count: int
    bearish_count: int
    unmitigated_count: int
    nearest: Optional[OrderBlock]
    distance_to_nearest: float


class OrderBlockEngine:
    """Finds displacement-validated order blocks and their lifecycle state."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def analyze(self, df: pd.DataFrame) -> OrderBlockState:
        """Detect order blocks across the series (vectorized candidate scan)."""
        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        n = len(df)
        atr = wilder_smooth(true_range(h, lo, c), self.config.atr_period)
        atr_fallback = float(np.median(h - lo)) or 1.0
        atr = np.where(np.isfinite(atr) & (atr > 0), atr, atr_fallback)
        timestamps = df["timestamp"].to_numpy()
        horizon = 3  # displacement must complete within this many candles

        # Forward-looking extremes over the next `horizon` bars.
        fwd_max = pd.Series(h[::-1]).rolling(horizon, min_periods=1).max().to_numpy()[::-1]
        fwd_min = pd.Series(lo[::-1]).rolling(horizon, min_periods=1).min().to_numpy()[::-1]
        fwd_max = np.roll(fwd_max, -1)
        fwd_min = np.roll(fwd_min, -1)
        fwd_max[-1] = h[-1]
        fwd_min[-1] = lo[-1]

        need = atr * self.config.ob_displacement_atr_multiple
        bearish_candle = c < o
        bullish_candle = c > o
        bull_ob = bearish_candle & ((fwd_max - h) >= need)
        bear_ob = bullish_candle & ((lo - fwd_min) >= need)
        # Keep only the last opposite candle before the impulse: drop candidates
        # immediately followed by another candidate of the same type.
        bull_ob[:-1] &= ~bull_ob[1:]
        bear_ob[:-1] &= ~bear_ob[1:]

        max_age = self.config.ob_max_age
        last_close = float(c[-1])
        atr_now = last_valid(atr, atr_fallback)

        blocks: List[OrderBlock] = []
        for direction, mask in (("bullish", bull_ob), ("bearish", bear_ob)):
            for i in np.nonzero(mask)[0]:
                i = int(i)
                age = n - 1 - i
                if age > max_age:
                    continue
                upper, lower = float(h[i]), float(lo[i])
                seg_lo = lo[i + horizon + 1:] if i + horizon + 1 < n else np.empty(0)
                seg_hi = h[i + horizon + 1:] if i + horizon + 1 < n else np.empty(0)
                if direction == "bullish":
                    entries = int(np.count_nonzero((seg_lo <= upper) & (seg_lo > lower)))
                    mitigated = bool(seg_lo.size and seg_lo.min() <= lower)
                    strength = float((fwd_max[i] - upper) / atr[i])
                    distance = (last_close - upper) / atr_now
                else:
                    entries = int(np.count_nonzero((seg_hi >= lower) & (seg_hi < upper)))
                    mitigated = bool(seg_hi.size and seg_hi.max() >= upper)
                    strength = float((lower - fwd_min[i]) / atr[i])
                    distance = (lower - last_close) / atr_now
                blocks.append(
                    OrderBlock(
                        direction=direction,  # type: ignore[arg-type]
                        index=i,
                        upper=upper,
                        lower=lower,
                        strength=strength,
                        age=age,
                        freshness=float(1.0 / (1 + entries)),
                        mitigated=mitigated,
                        retested=entries,
                        distance=float(distance),
                        timestamp=pd.Timestamp(timestamps[i]),
                    )
                )
        blocks.sort(key=lambda b: b.index)

        unmitigated = [b for b in blocks if not b.mitigated]
        nearest = min(unmitigated, key=lambda b: abs(b.distance), default=None)
        return OrderBlockState(
            blocks=blocks,
            bullish_count=sum(b.direction == "bullish" for b in blocks),
            bearish_count=sum(b.direction == "bearish" for b in blocks),
            unmitigated_count=len(unmitigated),
            nearest=nearest,
            distance_to_nearest=float(abs(nearest.distance)) if nearest else 0.0,
        )
