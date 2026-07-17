"""Trend classification from swing structure.

Classifies the market as bullish / bearish / sideways by scoring the sequence
of Higher Highs, Higher Lows, Lower Highs and Lower Lows over the most recent
swings, and derives strength, duration and momentum metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List

import numpy as np
import pandas as pd

from .config import EngineConfig
from .swings import SwingPoint, split_swings


class TrendDirection(IntEnum):
    """Numeric trend encoding suitable for ML feature vectors."""

    BEARISH = -1
    SIDEWAYS = 0
    BULLISH = 1


@dataclass(slots=True)
class TrendState:
    """Aggregate description of the prevailing trend.

    Attributes
    ----------
    direction:
        ``TrendDirection`` (-1 bearish, 0 sideways, +1 bullish).
    higher_highs, higher_lows, lower_highs, lower_lows:
        Counts over the configured swing lookback.
    strength:
        Net directional score in [0, 1]; 1 = perfectly one-sided structure.
    duration_bars:
        Bars since the current directional structure began.
    momentum:
        ATR-agnostic slope of recent swing prices per bar, as a fraction of
        the last close (e.g. 0.0001 = 1 pip per bar on a 1.0000 quote).
    valid:
        False when there weren't enough swings to score a direction (all
        count/strength/duration/momentum fields are placeholder defaults in
        that case, not a genuine "sideways" reading). Check this before
        trusting the other fields -- see FEATURE_OPTIMIZATION_REPORT.md,
        Task 3.
    """

    direction: TrendDirection
    higher_highs: int
    higher_lows: int
    lower_highs: int
    lower_lows: int
    strength: float
    duration_bars: int
    momentum: float
    valid: bool = False


class TrendEngine:
    """Derives :class:`TrendState` from detected swings."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def analyze(self, df: pd.DataFrame, swings: List[SwingPoint]) -> TrendState:
        """Classify the trend using the last ``trend_swing_lookback`` swings."""
        highs, lows = split_swings(swings)
        lb = self.config.trend_swing_lookback
        recent_highs = highs[-lb:]
        recent_lows = lows[-lb:]

        hh = _count_moves(recent_highs, up=True)
        lh = _count_moves(recent_highs, up=False)
        hl = _count_moves(recent_lows, up=True)
        ll = _count_moves(recent_lows, up=False)

        total = hh + lh + hl + ll
        score = ((hh + hl) - (lh + ll)) / total if total else 0.0

        if score > self.config.sideways_threshold:
            direction = TrendDirection.BULLISH
        elif score < -self.config.sideways_threshold:
            direction = TrendDirection.BEARISH
        else:
            direction = TrendDirection.SIDEWAYS

        duration = _trend_duration(df, swings, direction)
        momentum = _swing_momentum(df, recent_highs, recent_lows)

        return TrendState(
            direction=direction,
            higher_highs=hh,
            higher_lows=hl,
            lower_highs=lh,
            lower_lows=ll,
            strength=abs(score),
            duration_bars=duration,
            momentum=momentum,
            valid=total > 0,
        )


def _count_moves(points: List[SwingPoint], up: bool) -> int:
    """Count consecutive-pair moves up (or down) within a swing sequence."""
    if len(points) < 2:
        return 0
    prices = np.array([p.price for p in points])
    diffs = np.diff(prices)
    return int((diffs > 0).sum()) if up else int((diffs < 0).sum())


def _trend_duration(
    df: pd.DataFrame, swings: List[SwingPoint], direction: TrendDirection
) -> int:
    """Bars since the swing where the current directional leg began."""
    if not swings:
        return 0
    last_bar = len(df) - 1
    if direction == TrendDirection.SIDEWAYS:
        return last_bar - swings[-1].index
    highs, lows = split_swings(swings)
    anchor = swings[0].index
    seq = highs if direction == TrendDirection.BULLISH else lows
    if len(seq) >= 2:
        prices = [p.price for p in seq]
        for i in range(len(seq) - 1, 0, -1):
            rising = prices[i] > prices[i - 1]
            if (direction == TrendDirection.BULLISH) != rising:
                anchor = seq[i].index
                break
        else:
            anchor = seq[0].index
    return max(last_bar - anchor, 0)


def _swing_momentum(
    df: pd.DataFrame, highs: List[SwingPoint], lows: List[SwingPoint]
) -> float:
    """Least-squares slope of recent swing prices per bar, close-normalized."""
    pts = sorted(highs + lows, key=lambda s: s.index)
    if len(pts) < 2:
        return 0.0
    x = np.array([p.index for p in pts], dtype=float)
    y = np.array([p.price for p in pts], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    last_close = float(df["close"].iloc[-1])
    return slope / last_close if last_close else 0.0
