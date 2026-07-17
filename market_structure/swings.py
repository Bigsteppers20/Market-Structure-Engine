"""Swing (pivot) detection.

Detects swing highs and swing lows with a symmetric dominance window, fully
vectorized via stride-free rolling max/min comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import pandas as pd

from .config import EngineConfig
from .utils import safe_divide, true_range, wilder_smooth

SwingKind = Literal["high", "low"]


@dataclass(slots=True)
class SwingPoint:
    """A confirmed swing high or swing low.

    Attributes
    ----------
    index:
        Integer bar index of the pivot.
    price:
        Pivot price (high for swing highs, low for swing lows).
    timestamp:
        Bar timestamp of the pivot.
    kind:
        ``"high"`` or ``"low"``.
    strength:
        ATR-normalized prominence of the pivot over its neighbors (>= 0).
    distance_from_previous:
        Bars since the previous swing of the same kind (0 for the first).
    """

    index: int
    price: float
    timestamp: pd.Timestamp
    kind: SwingKind
    strength: float
    distance_from_previous: int


class SwingDetector:
    """Detects swing highs/lows with an ``EngineConfig.swing_window`` pivot rule."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def detect(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Return all confirmed swing points sorted by bar index."""
        k = self.config.swing_window
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        n = len(df)
        if n < 2 * k + 1:
            return []

        atr = wilder_smooth(true_range(high, low, close), self.config.atr_period)
        atr = np.where(np.isfinite(atr) & (atr > 0), atr, np.nanmedian(high - low) or 1.0)

        window = 2 * k + 1
        roll_max = pd.Series(high).rolling(window, center=True).max().to_numpy()
        roll_min = pd.Series(low).rolling(window, center=True).min().to_numpy()

        is_high = (high >= roll_max) & np.isfinite(roll_max)
        is_low = (low <= roll_min) & np.isfinite(roll_min)
        # Break ties among equal plateau highs/lows: keep the first bar only.
        is_high &= ~self._plateau_repeat(high, is_high)
        is_low &= ~self._plateau_repeat(low, is_low)

        timestamps = df["timestamp"].to_numpy()
        swings: List[SwingPoint] = []
        for kind, mask, prices in (("high", is_high, high), ("low", is_low, low)):
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                continue
            # Prominence = pivot price vs. the best neighbor on either side,
            # computed with two shifted rolling extremes (fully vectorized).
            s = pd.Series(prices)
            if kind == "high":
                left = s.rolling(k, min_periods=1).max().shift(1).to_numpy()
                right = s[::-1].rolling(k, min_periods=1).max().shift(1).to_numpy()[::-1]
                neigh = np.fmax(left, right)
                prom = prices[idxs] - neigh[idxs]
            else:
                left = s.rolling(k, min_periods=1).min().shift(1).to_numpy()
                right = s[::-1].rolling(k, min_periods=1).min().shift(1).to_numpy()[::-1]
                neigh = np.fmin(left, right)
                prom = neigh[idxs] - prices[idxs]
            prom = np.where(np.isfinite(prom), prom, 0.0)
            strengths = safe_divide(np.maximum(prom, 0.0), atr[idxs])
            prev = np.concatenate(([0], np.diff(idxs)))
            for j, i in enumerate(idxs):
                swings.append(
                    SwingPoint(
                        index=int(i),
                        price=float(prices[i]),
                        timestamp=pd.Timestamp(timestamps[i]),
                        kind=kind,  # type: ignore[arg-type]
                        strength=float(strengths[j]),
                        distance_from_previous=int(prev[j]),
                    )
                )
        swings.sort(key=lambda s: s.index)
        return swings

    @staticmethod
    def _plateau_repeat(prices: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """True for pivot candidates whose price equals the previous candidate's."""
        repeat = np.zeros_like(mask)
        idxs = np.nonzero(mask)[0]
        if idxs.size < 2:
            return repeat
        same = prices[idxs[1:]] == prices[idxs[:-1]]
        close_together = np.diff(idxs) <= 2
        repeat[idxs[1:][same & close_together]] = True
        return repeat


def split_swings(swings: List[SwingPoint]) -> tuple[List[SwingPoint], List[SwingPoint]]:
    """Split a mixed swing list into (highs, lows), each sorted by index."""
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    return highs, lows
