"""Support and resistance zones.

Builds price zones by clustering swing lows (support) and swing highs
(resistance) that sit within an ATR-scaled tolerance of each other, then
counts touches and scores zone strength.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from .config import EngineConfig
from .swings import SwingPoint, split_swings
from .utils import last_valid, true_range, wilder_smooth

ZoneKind = Literal["support", "resistance"]


@dataclass(slots=True)
class Zone:
    """A horizontal support or resistance zone.

    Attributes
    ----------
    kind:
        ``"support"`` or ``"resistance"``.
    lower, upper:
        Zone price bounds.
    center:
        Midpoint of the zone.
    touches:
        Number of swing points that formed / retested the zone.
    strength:
        Composite score: touches weighted by swing strength and recency.
    width:
        ``upper - lower`` in price units.
    last_touch_index:
        Bar index of the most recent touch.
    """

    kind: ZoneKind
    lower: float
    upper: float
    center: float
    touches: int
    strength: float
    width: float
    last_touch_index: int


@dataclass(slots=True)
class ZoneSummary:
    """Zones plus distances from the current close."""

    support_zones: List[Zone]
    resistance_zones: List[Zone]
    nearest_support: Optional[Zone]
    nearest_resistance: Optional[Zone]
    distance_to_support: float
    """Close minus nearest support center, ATR-normalized (>=0 typical)."""
    distance_to_resistance: float
    """Nearest resistance center minus close, ATR-normalized (>=0 typical)."""


class SupportResistanceEngine:
    """Clusters swings into zones and measures proximity."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def analyze(self, df: pd.DataFrame, swings: List[SwingPoint]) -> ZoneSummary:
        """Build support/resistance zones and distances to the last close."""
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        atr_now = last_valid(
            wilder_smooth(true_range(high, low, close), self.config.atr_period),
            default=float(np.median(high - low)) or 1.0,
        )
        tol = atr_now * self.config.zone_merge_atr_multiple
        n = len(df)

        highs, lows = split_swings(swings)
        resistance = self._cluster(highs, "resistance", tol, n)
        support = self._cluster(lows, "support", tol, n)
        resistance = sorted(resistance, key=lambda z: z.strength, reverse=True)[
            : self.config.max_zones
        ]
        support = sorted(support, key=lambda z: z.strength, reverse=True)[
            : self.config.max_zones
        ]

        last_close = float(close[-1])
        below = [z for z in support if z.center <= last_close] or support
        above = [z for z in resistance if z.center >= last_close] or resistance
        nearest_sup = min(below, key=lambda z: abs(last_close - z.center)) if below else None
        nearest_res = min(above, key=lambda z: abs(z.center - last_close)) if above else None
        d_sup = (last_close - nearest_sup.center) / atr_now if nearest_sup else 0.0
        d_res = (nearest_res.center - last_close) / atr_now if nearest_res else 0.0

        return ZoneSummary(
            support_zones=support,
            resistance_zones=resistance,
            nearest_support=nearest_sup,
            nearest_resistance=nearest_res,
            distance_to_support=float(d_sup),
            distance_to_resistance=float(d_res),
        )

    @staticmethod
    def _cluster(points: List[SwingPoint], kind: ZoneKind, tol: float, n_bars: int) -> List[Zone]:
        """Single-linkage 1-D clustering of swing prices within ``tol``."""
        if not points:
            return []
        pts = sorted(points, key=lambda p: p.price)
        zones: List[Zone] = []
        cluster: List[SwingPoint] = [pts[0]]
        for p in pts[1:]:
            if p.price - cluster[-1].price <= tol:
                cluster.append(p)
            else:
                zones.append(SupportResistanceEngine._to_zone(cluster, kind, n_bars))
                cluster = [p]
        zones.append(SupportResistanceEngine._to_zone(cluster, kind, n_bars))
        return zones

    @staticmethod
    def _to_zone(cluster: List[SwingPoint], kind: ZoneKind, n_bars: int) -> Zone:
        prices = np.array([p.price for p in cluster])
        last_idx = max(p.index for p in cluster)
        recency = 0.5 + 0.5 * (last_idx / max(n_bars - 1, 1))
        strength = float((len(cluster) + sum(p.strength for p in cluster) * 0.25) * recency)
        lower, upper = float(prices.min()), float(prices.max())
        return Zone(
            kind=kind,
            lower=lower,
            upper=upper,
            center=float(prices.mean()),
            touches=len(cluster),
            strength=strength,
            width=upper - lower,
            last_touch_index=last_idx,
        )
