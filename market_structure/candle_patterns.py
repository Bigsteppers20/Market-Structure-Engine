"""Candlestick pattern recognition, fully vectorized.

Every pattern is computed as a boolean array over the whole series and
encoded numerically (0/1, or signed for two-sided patterns) for direct use
in ML feature vectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from .config import EngineConfig


_EPS = 1e-12


@dataclass(slots=True)
class PatternPanel:
    """Boolean pattern arrays plus the last-bar numeric encoding.

    Attributes
    ----------
    series:
        Mapping pattern name -> ``np.ndarray`` of 0/1 flags (full length).
    snapshot:
        Mapping pattern name -> value on the most recent candle.

    Note
    ----
    There is no ``bullish_score``/``bearish_score`` composite: it was an
    exact sum of the bullish/bearish flags below and carried no information
    beyond them (removed per FEATURE_OPTIMIZATION_REPORT.md, Task 2). A
    downstream model can reconstruct any weighted combination it needs from
    the individual flags.
    """

    series: Dict[str, np.ndarray] = field(default_factory=dict)
    snapshot: Dict[str, float] = field(default_factory=dict)


class CandlePatternEngine:
    """Detects 20+ classic candlestick patterns without loops."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def analyze(self, df: pd.DataFrame) -> PatternPanel:
        """Compute all pattern flags for every candle."""
        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        n = len(df)

        body = np.abs(c - o)
        rng = np.maximum(h - lo, _EPS)
        upper = h - np.maximum(o, c)
        lower = np.minimum(o, c) - lo
        bull = c > o
        bear = c < o
        body_ratio = body / rng
        avg_body = pd.Series(body).rolling(14, min_periods=1).mean().to_numpy()
        small_body = body <= 0.35 * rng
        tiny_body = body <= 0.1 * rng
        long_body = body >= 0.6 * rng

        def shift(a: np.ndarray, k: int = 1) -> np.ndarray:
            out = np.empty_like(a)
            out[:k] = np.nan
            out[k:] = a[:-k]
            return out

        o1, c1, h1, l1 = shift(o), shift(c), shift(h), shift(lo)
        o2, c2 = shift(o, 2), shift(c, 2)
        body1, body2 = np.abs(c1 - o1), np.abs(c2 - o2)
        bull1, bear1 = c1 > o1, c1 < o1
        bull2, bear2 = c2 > o2, c2 < o2

        p: Dict[str, np.ndarray] = {}
        p["doji"] = tiny_body
        p["dragonfly_doji"] = tiny_body & (lower >= 0.6 * rng) & (upper <= 0.1 * rng)
        p["gravestone_doji"] = tiny_body & (upper >= 0.6 * rng) & (lower <= 0.1 * rng)
        p["hammer"] = small_body & (lower >= 2.0 * np.maximum(body, _EPS)
                                    ) & (upper <= 0.3 * body + 0.1 * rng)
        p["inverted_hammer"] = small_body & (
            upper >= 2.0 * np.maximum(body, _EPS)) & (lower <= 0.3 * body + 0.1 * rng)
        p["shooting_star"] = p["inverted_hammer"] & bull1
        p["bullish_engulfing"] = bull & bear1 & (c >= o1) & (o <= c1) & (body > body1)
        p["bearish_engulfing"] = bear & bull1 & (c <= o1) & (o >= c1) & (body > body1)
        p["bullish_harami"] = bull & bear1 & (o >= c1) & (c <= o1) & (body < body1)
        p["bearish_harami"] = bear & bull1 & (o <= c1) & (c >= o1) & (body < body1)
        # No `harami` union: exactly (bullish_harami | bearish_harami), zero
        # information beyond those two flags (Task 2: redundant features).
        mid1 = (o1 + c1) / 2.0
        p["piercing_line"] = bull & bear1 & (o < l1) & (c > mid1) & (c < o1)
        p["dark_cloud_cover"] = bear & bull1 & (o > h1) & (c < mid1) & (c > o1)
        p["morning_star"] = (
            bear2 & (body2 > avg_body) & (np.abs(c1 - o1) < 0.3 * body2)
            & bull & (c > (o2 + c2) / 2.0)
        )
        p["evening_star"] = (
            bull2 & (body2 > avg_body) & (np.abs(c1 - o1) < 0.3 * body2)
            & bear & (c < (o2 + c2) / 2.0)
        )
        p["three_white_soldiers"] = bull & bull1 & bull2 & (c > c1) & (c1 > c2) & long_body
        p["three_black_crows"] = bear & bear1 & bear2 & (c < c1) & (c1 < c2) & long_body
        p["inside_bar"] = (h <= h1) & (lo >= l1)
        p["outside_bar"] = (h > h1) & (lo < l1)
        p["bullish_pin_bar"] = (lower >= 0.66 * rng) & (upper <= 0.15 * rng)
        p["bearish_pin_bar"] = (upper >= 0.66 * rng) & (lower <= 0.15 * rng)
        # No `pin_bar` union: exactly (bullish_pin_bar | bearish_pin_bar).
        p["bullish_marubozu"] = bull & (body_ratio >= 0.95)
        p["bearish_marubozu"] = bear & (body_ratio >= 0.95)
        # No `marubozu` union: exactly (bullish_marubozu | bearish_marubozu).
        p["spinning_top"] = small_body & (upper >= 0.25 * rng) & (lower >= 0.25 * rng) & ~tiny_body

        series = {k: np.nan_to_num(v.astype(float)) for k, v in p.items()}
        snapshot = {k: float(v[-1]) if n else 0.0 for k, v in series.items()}
        return PatternPanel(series=series, snapshot=snapshot)
