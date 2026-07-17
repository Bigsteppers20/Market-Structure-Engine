"""Spread-derived features from optional broker spread data.

``spread`` (ask-close minus bid-close) is validated by :class:`DataLoader` as
an optional OHLCV column but, before this module, was never read by any
feature. A wide or spiking spread is itself a liquidity/regime signal (thin
order books, news events, session transitions), so this engine derives a
small set of spread-regime features when the column is present -- and marks
them explicitly invalid (never a silent ``0.0``) when it is not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EngineConfig
from .utils import last_valid, true_range, wilder_smooth


@dataclass(slots=True)
class SpreadFeatures:
    """Spread regime features for the last candle.

    Attributes
    ----------
    current:
        Last candle's spread, in price units.
    rolling_avg:
        Mean spread over the trailing ``spread_window`` candles.
    spike:
        1.0 if ``current`` is at least ``spread_spike_multiple`` times
        ``rolling_avg``.
    atr_ratio:
        ``current`` divided by the current ATR (spread relative to typical
        candle range).
    percentile:
        Fraction of the trailing ``spread_window`` spreads at or below
        ``current``, in ``[0, 1]``.
    volatility:
        Rolling standard deviation of spread over ``spread_window`` candles.
    distance_from_avg:
        ``(current - rolling_avg) / rolling_avg``, a signed relative
        deviation from the recent average spread.
    valid:
        1.0 only when the input data provided a ``spread`` column AND at
        least ``spread_window`` candles are loaded. All other fields above
        are meaningless placeholders (0.0) when this is 0.0 -- never treat
        a 0.0 spread reading as real without checking this first.
    """

    current: float = 0.0
    rolling_avg: float = 0.0
    spike: float = 0.0
    atr_ratio: float = 0.0
    percentile: float = 0.0
    volatility: float = 0.0
    distance_from_avg: float = 0.0
    valid: float = 0.0


class SpreadEngine:
    """Derives spread-regime features from an optional ``spread`` column."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def analyze(self, df: pd.DataFrame) -> SpreadFeatures:
        """Compute spread features for the last loaded candle.

        Returns an all-zero, ``valid=0.0`` :class:`SpreadFeatures` when the
        input has no ``spread`` column or too little history -- callers must
        check ``valid`` before trusting any other field.
        """
        if "spread" not in df.columns:
            return SpreadFeatures()

        cfg = self.config
        spread = df["spread"].to_numpy(dtype=float)
        n = len(spread)
        w = cfg.spread_window
        current = float(spread[-1]) if n else float("nan")
        if n < w or not np.isfinite(current):
            return SpreadFeatures()

        s = pd.Series(spread)
        rolling_avg = s.rolling(w, min_periods=1).mean().to_numpy()
        rolling_std = s.rolling(w, min_periods=1).std(ddof=0).to_numpy()
        rolling_std = np.where(np.isfinite(rolling_std), rolling_std, 0.0)

        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        atr = wilder_smooth(true_range(high, low, close), cfg.atr_period)
        atr_fallback = float(np.median(high - low)) or 1.0
        atr_now = last_valid(atr, atr_fallback)

        avg_now = float(rolling_avg[-1])
        std_now = float(rolling_std[-1])
        window = spread[max(0, n - w):]
        percentile = float(np.mean(window <= current)) if window.size else 0.0

        return SpreadFeatures(
            current=current,
            rolling_avg=avg_now,
            spike=float(bool(avg_now > 0 and (current / avg_now) >= cfg.spread_spike_multiple)),
            atr_ratio=float(current / atr_now) if atr_now else 0.0,
            percentile=percentile,
            volatility=std_now,
            distance_from_avg=float((current - avg_now) / avg_now) if avg_now else 0.0,
            valid=1.0,
        )
