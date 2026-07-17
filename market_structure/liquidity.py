"""Liquidity analysis.

Detects equal highs / equal lows (resting liquidity pools), classifies buy-
side vs sell-side liquidity, and finds liquidity sweeps: wicks that pierce a
pool and close back inside.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from .config import EngineConfig
from .swings import SwingPoint, split_swings
from .utils import last_valid, true_range, wilder_smooth

PoolSide = Literal["buy_side", "sell_side"]
SweepDirection = Literal["above", "below"]


@dataclass(slots=True)
class LiquidityPool:
    """A cluster of equal highs (buy-side) or equal lows (sell-side).

    Attributes
    ----------
    side:
        ``"buy_side"`` (above equal highs) or ``"sell_side"`` (below equal lows).
    price:
        Mean price of the equal levels.
    touches:
        Number of swings forming the pool (>= 2).
    strength:
        Touches weighted by how tightly the levels align (tighter = stronger).
    first_index, last_index:
        Bar indices spanning the pool's formation.
    swept:
        True once price has traded through the pool.
    """

    side: PoolSide
    price: float
    touches: int
    strength: float
    first_index: int
    last_index: int
    swept: bool


@dataclass(slots=True)
class LiquiditySweep:
    """A single sweep event: a wick through a pool with a close back inside.

    Attributes
    ----------
    direction:
        ``"above"`` = buy-side liquidity taken; ``"below"`` = sell-side.
    index:
        Bar index of the sweeping candle.
    pool_price:
        The pool level that was swept.
    size:
        ATR-normalized wick penetration beyond the pool.
    timestamp:
        Timestamp of the sweeping candle.
    """

    direction: SweepDirection
    index: int
    pool_price: float
    size: float
    timestamp: pd.Timestamp


@dataclass(slots=True)
class LiquidityState:
    """Aggregate liquidity picture for the feature vector."""

    pools: List[LiquidityPool]
    sweeps: List[LiquiditySweep]
    equal_highs: int
    equal_lows: int
    buy_side_liquidity: float
    """Total strength of unswept buy-side pools above price."""
    sell_side_liquidity: float
    """Total strength of unswept sell-side pools below price."""
    last_sweep: Optional[LiquiditySweep]


class LiquidityEngine:
    """Builds pools from swing clusters and scans candles for sweeps."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def analyze(self, df: pd.DataFrame, swings: List[SwingPoint]) -> LiquidityState:
        """Detect pools and sweeps over the full series."""
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        atr_now = last_valid(
            wilder_smooth(true_range(high, low, close), self.config.atr_period),
            default=float(np.median(high - low)) or 1.0,
        )
        tol = atr_now * self.config.equal_level_atr_tolerance

        highs, lows = split_swings(swings)
        pools = self._pools(highs, "buy_side", tol) + self._pools(lows, "sell_side", tol)

        sweeps: List[LiquiditySweep] = []
        timestamps = df["timestamp"].to_numpy()
        ratio = self.config.sweep_close_back_ratio
        for pool in pools:
            start = pool.last_index + 1
            if start >= len(df):
                continue
            if pool.side == "buy_side":
                pierced = np.nonzero(high[start:] > pool.price)[0]
            else:
                pierced = np.nonzero(low[start:] < pool.price)[0]
            if pierced.size == 0:
                continue
            j = start + int(pierced[0])
            pool.swept = True
            if pool.side == "buy_side":
                penetration = high[j] - pool.price
                closed_back = close[j] <= pool.price + (1 - ratio) * penetration
                direction: SweepDirection = "above"
            else:
                penetration = pool.price - low[j]
                closed_back = close[j] >= pool.price - (1 - ratio) * penetration
                direction = "below"
            if closed_back and penetration > 0:
                sweeps.append(
                    LiquiditySweep(
                        direction=direction,
                        index=j,
                        pool_price=pool.price,
                        size=float(penetration / atr_now),
                        timestamp=pd.Timestamp(timestamps[j]),
                    )
                )
        sweeps.sort(key=lambda s: s.index)

        last_close = float(close[-1])
        bsl = sum(p.strength for p in pools if p.side ==
                  "buy_side" and not p.swept and p.price > last_close)
        ssl = sum(p.strength for p in pools if p.side ==
                  "sell_side" and not p.swept and p.price < last_close)

        return LiquidityState(
            pools=pools,
            sweeps=sweeps,
            equal_highs=sum(p.touches for p in pools if p.side == "buy_side"),
            equal_lows=sum(p.touches for p in pools if p.side == "sell_side"),
            buy_side_liquidity=float(bsl),
            sell_side_liquidity=float(ssl),
            last_sweep=sweeps[-1] if sweeps else None,
        )

    @staticmethod
    def _pools(points: List[SwingPoint], side: PoolSide, tol: float) -> List[LiquidityPool]:
        """Cluster same-kind swings within ``tol`` into pools of >= 2 touches."""
        if len(points) < 2:
            return []
        pts = sorted(points, key=lambda p: p.price)
        pools: List[LiquidityPool] = []
        cluster: List[SwingPoint] = [pts[0]]
        for p in pts[1:]:
            if p.price - cluster[-1].price <= tol:
                cluster.append(p)
            else:
                if len(cluster) >= 2:
                    pools.append(LiquidityEngine._to_pool(cluster, side, tol))
                cluster = [p]
        if len(cluster) >= 2:
            pools.append(LiquidityEngine._to_pool(cluster, side, tol))
        return pools

    @staticmethod
    def _to_pool(cluster: List[SwingPoint], side: PoolSide, tol: float) -> LiquidityPool:
        prices = np.array([p.price for p in cluster])
        spread = float(prices.max() - prices.min())
        tightness = 1.0 - min(spread / tol, 1.0) * 0.5 if tol > 0 else 1.0
        return LiquidityPool(
            side=side,
            price=float(prices.mean()),
            touches=len(cluster),
            strength=float(len(cluster) * tightness),
            first_index=min(p.index for p in cluster),
            last_index=max(p.index for p in cluster),
            swept=False,
        )
