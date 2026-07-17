"""MarketState assembly and ML feature-vector flattening.

Builds price-action, volatility, microstructure and session features, then
combines them with every sub-engine's output into a single
:class:`MarketState` dataclass. ``MarketState.to_vector()`` produces a flat,
deterministically ordered numeric vector for downstream models.

This module never emits BUY/SELL/NO_TRADE — features only.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .bos import StructureBreak

from .choch import ChochEvent
from .config import EngineConfig
from .fvg import FvgState
from .indicators import IndicatorPanel
from .liquidity import LiquidityState
from .order_blocks import OrderBlockState
from .spread import SpreadFeatures
from .support_resistance import ZoneSummary
from .swings import SwingPoint
from .trend import TrendState
from .utils import scalar


# --------------------------------------------------------------------------- #
# feature groups
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PriceActionFeatures:
    """Per-bar price action metrics for the most recent candle."""

    current_close: float = 0.0
    previous_close: float = 0.0
    price_change: float = 0.0
    price_return: float = 0.0
    body_size: float = 0.0
    upper_wick: float = 0.0
    lower_wick: float = 0.0
    body_ratio: float = 0.0
    candle_range: float = 0.0
    atr_ratio: float = 0.0
    gap_up: float = 0.0
    gap_down: float = 0.0
    distance_from_ema20: float = 0.0
    distance_from_ema50: float = 0.0
    distance_from_ema200: float = 0.0
    distance_from_vwap: float = 0.0


@dataclass(slots=True)
class VolatilityFeatures:
    """Volatility regime description.

    Note
    ----
    ``atr``, ``true_range`` and ``rolling_volatility`` were removed as exact
    duplicates of ``ind_atr``, ``ind_true_range`` and ``ind_volatility``
    (same underlying :class:`IndicatorPanel` snapshot values) -- see
    FEATURE_OPTIMIZATION_REPORT.md, Task 1.
    """

    historical_volatility: float = 0.0
    average_candle_size: float = 0.0
    average_wick_size: float = 0.0
    expansion: float = 0.0
    compression: float = 0.0
    valid: float = 0.0
    """1.0 once >= hist_vol_window candles are loaded; 0.0 = placeholder."""


@dataclass(slots=True)
class MicrostructureFeatures:
    """Impulse/correction geometry derived from swings."""

    impulse_length: float = 0.0
    correction_length: float = 0.0
    impulse_ratio: float = 0.0
    retracement_pct: float = 0.0
    extension_pct: float = 0.0
    swing_velocity: float = 0.0
    swing_acceleration: float = 0.0
    time_between_swings: float = 0.0
    average_swing_length: float = 0.0
    valid: float = 0.0
    """1.0 once >= 3 swings are available to derive leg geometry; 0.0 = placeholder."""


@dataclass(slots=True)
class SessionFeatures:
    """Trading-session and calendar encodings (UTC-based).

    Note
    ----
    ``session_overlap`` was removed: it was an exact deterministic function
    of the 4 ``is_*`` flags (>= 2 active), carrying no information beyond
    them -- see FEATURE_OPTIMIZATION_REPORT.md, Task 2.
    """

    is_sydney: float = 0.0
    is_asian: float = 0.0
    is_london: float = 0.0
    is_newyork: float = 0.0
    hour: float = 0.0
    minute: float = 0.0
    day_of_week: float = 0.0
    month: float = 0.0


@dataclass(slots=True)
class StructureFeatures:
    """BOS / CHOCH numeric summary."""

    last_bos_direction: float = 0.0
    last_bos_strength: float = 0.0
    bars_since_bos: float = -1.0
    bullish_bos_count: float = 0.0
    bearish_bos_count: float = 0.0
    last_choch_direction: float = 0.0
    last_choch_strength: float = 0.0
    bars_since_choch: float = -1.0
    choch_count: float = 0.0


# --------------------------------------------------------------------------- #
# MarketState
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class MarketState:
    """Complete numerical description of the current market state.

    Contains only descriptive features — never trade decisions.
    """

    n_candles: int = 0
    trend: TrendState | None = None
    structure: StructureFeatures = field(default_factory=StructureFeatures)
    price_action: PriceActionFeatures = field(default_factory=PriceActionFeatures)
    volatility: VolatilityFeatures = field(default_factory=VolatilityFeatures)
    microstructure: MicrostructureFeatures = field(default_factory=MicrostructureFeatures)
    session: SessionFeatures = field(default_factory=SessionFeatures)
    zones: ZoneSummary | None = None
    liquidity: LiquidityState | None = None
    fvg: FvgState | None = None
    order_blocks: OrderBlockState | None = None
    spread: SpreadFeatures = field(default_factory=SpreadFeatures)
    indicators: Dict[str, float] = field(default_factory=dict)
    indicator_validity: Dict[str, float] = field(default_factory=dict)
    patterns: Dict[str, float] = field(default_factory=dict)
    swings: List[SwingPoint] = field(default_factory=list)
    breaks: List[StructureBreak] = field(default_factory=list)
    chochs: List[ChochEvent] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, float]:
        """Flatten every numeric feature into an ordered name -> value map."""
        out: Dict[str, float] = {"n_candles": float(self.n_candles)}
        if self.trend is not None:
            t = self.trend
            out.update(
                trend_direction=float(int(t.direction)),
                trend_higher_highs=float(t.higher_highs),
                trend_higher_lows=float(t.higher_lows),
                trend_lower_highs=float(t.lower_highs),
                trend_lower_lows=float(t.lower_lows),
                trend_strength=t.strength,
                trend_duration_bars=float(t.duration_bars),
                trend_momentum=t.momentum,
                trend_valid=float(t.valid),
            )
        for group, prefix in (
            (self.structure, "structure"),
            (self.price_action, "pa"),
            (self.volatility, "vol"),
            (self.microstructure, "micro"),
            (self.session, "session"),
            (self.spread, "spread"),
        ):
            for f in fields(group):
                out[f"{prefix}_{f.name}"] = scalar(getattr(group, f.name))
        if self.zones is not None:
            z = self.zones
            out.update(
                sr_support_zone_count=float(len(z.support_zones)),
                sr_resistance_zone_count=float(len(z.resistance_zones)),
                sr_nearest_support_strength=(
                    z.nearest_support.strength if z.nearest_support else 0.0),
                sr_nearest_support_touches=float(
                    z.nearest_support.touches) if z.nearest_support else 0.0,
                sr_nearest_support_width=z.nearest_support.width if z.nearest_support else 0.0,
                sr_nearest_resistance_strength=(
                    z.nearest_resistance.strength if z.nearest_resistance else 0.0),
                sr_nearest_resistance_touches=float(
                    z.nearest_resistance.touches) if z.nearest_resistance else 0.0,
                sr_nearest_resistance_width=(
                    z.nearest_resistance.width if z.nearest_resistance else 0.0),
                sr_distance_to_support=z.distance_to_support,
                sr_distance_to_resistance=z.distance_to_resistance,
                sr_support_valid=1.0 if z.nearest_support else 0.0,
                sr_resistance_valid=1.0 if z.nearest_resistance else 0.0,
            )
        if self.liquidity is not None:
            lq = self.liquidity
            out.update(
                liq_equal_highs=float(lq.equal_highs),
                liq_equal_lows=float(lq.equal_lows),
                liq_pool_count=float(len(lq.pools)),
                liq_sweep_count=float(len(lq.sweeps)),
                liq_buy_side=lq.buy_side_liquidity,
                liq_sell_side=lq.sell_side_liquidity,
                liq_last_sweep_direction=(
                    1.0 if lq.last_sweep and lq.last_sweep.direction == "above"
                    else -1.0 if lq.last_sweep else 0.0
                ),
                liq_last_sweep_size=lq.last_sweep.size if lq.last_sweep else 0.0,
            )
        if self.fvg is not None:
            g = self.fvg
            out.update(
                fvg_bullish_count=float(g.bullish_count),
                fvg_bearish_count=float(g.bearish_count),
                fvg_unfilled_count=float(g.unfilled_count),
                fvg_nearest_direction=(
                    1.0 if g.nearest and g.nearest.direction == "bullish"
                    else -1.0 if g.nearest else 0.0
                ),
                fvg_nearest_size_atr=g.nearest.size_atr if g.nearest else 0.0,
                fvg_nearest_age=float(g.nearest.age) if g.nearest else 0.0,
                fvg_distance_to_nearest=g.distance_to_nearest,
            )
        if self.order_blocks is not None:
            ob = self.order_blocks
            out.update(
                ob_bullish_count=float(ob.bullish_count),
                ob_bearish_count=float(ob.bearish_count),
                ob_unmitigated_count=float(ob.unmitigated_count),
                ob_nearest_direction=(
                    1.0 if ob.nearest and ob.nearest.direction == "bullish"
                    else -1.0 if ob.nearest else 0.0
                ),
                ob_nearest_strength=ob.nearest.strength if ob.nearest else 0.0,
                ob_nearest_freshness=ob.nearest.freshness if ob.nearest else 0.0,
                ob_nearest_age=float(ob.nearest.age) if ob.nearest else 0.0,
                ob_distance_to_nearest=ob.distance_to_nearest,
            )
        for name in sorted(self.indicators):
            out[f"ind_{name}"] = scalar(self.indicators[name])
            out[f"ind_{name}_valid"] = scalar(self.indicator_validity.get(name, 0.0))
        for name in sorted(self.patterns):
            out[f"pat_{name}"] = scalar(self.patterns[name])
        return out

    def to_vector(self) -> Tuple[np.ndarray, List[str]]:
        """Return ``(values, feature_names)`` as a flat float64 vector."""
        d = self.to_dict()
        names = list(d.keys())
        return np.array([d[k] for k in names], dtype=np.float64), names


# --------------------------------------------------------------------------- #
# feature builders
# --------------------------------------------------------------------------- #
def build_price_action(df: pd.DataFrame, panel: IndicatorPanel) -> PriceActionFeatures:
    """Compute last-candle price action features from the indicator panel."""
    o = float(df["open"].iloc[-1])
    h = float(df["high"].iloc[-1])
    lo = float(df["low"].iloc[-1])
    c = float(df["close"].iloc[-1])
    prev_c = float(df["close"].iloc[-2]) if len(df) > 1 else c
    prev_h = float(df["high"].iloc[-2]) if len(df) > 1 else h
    prev_l = float(df["low"].iloc[-2]) if len(df) > 1 else lo
    body = abs(c - o)
    rng = max(h - lo, 1e-12)
    atr_now = max(panel.snapshot.get("atr", 0.0), 1e-12)
    snap = panel.snapshot
    return PriceActionFeatures(
        current_close=c,
        previous_close=prev_c,
        price_change=c - prev_c,
        price_return=(c - prev_c) / prev_c if prev_c else 0.0,
        body_size=body,
        upper_wick=h - max(o, c),
        lower_wick=min(o, c) - lo,
        body_ratio=body / rng,
        candle_range=h - lo,
        atr_ratio=(h - lo) / atr_now,
        gap_up=float(min(o, lo) > prev_h),
        gap_down=float(max(o, h) < prev_l),
        distance_from_ema20=(c - snap.get("ema_20", c)) / atr_now,
        distance_from_ema50=(c - snap.get("ema_50", c)) / atr_now,
        distance_from_ema200=(c - snap.get("ema_200", c)) / atr_now,
        distance_from_vwap=(c - snap.get("vwap", c)) / atr_now,
    )


def build_volatility(
    df: pd.DataFrame, panel: IndicatorPanel, config: EngineConfig
) -> VolatilityFeatures:
    """Compute volatility regime features."""
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    w = config.hist_vol_window
    returns = np.diff(np.log(np.maximum(c, 1e-12)))
    hist_vol = float(np.std(returns[-w:], ddof=0)) if returns.size >= 2 else 0.0
    atr_series = panel.series["atr"]
    atr_now = panel.snapshot.get("atr", 0.0)
    finite_atr = atr_series[np.isfinite(atr_series)]
    atr_mean = float(finite_atr[-w:].mean()) if finite_atr.size else 0.0
    ratio = atr_now / atr_mean if atr_mean else 1.0
    wick = (h - np.maximum(o, c)) + (np.minimum(o, c) - lo)
    return VolatilityFeatures(
        historical_volatility=hist_vol,
        average_candle_size=float((h - lo)[-w:].mean()),
        average_wick_size=float(wick[-w:].mean()),
        expansion=float(ratio >= config.expansion_ratio),
        compression=float(ratio <= config.compression_ratio),
        valid=float(len(df) >= w),
    )


def build_microstructure(
    df: pd.DataFrame, swings: List[SwingPoint], trend: TrendState
) -> MicrostructureFeatures:
    """Compute impulse/correction geometry from the last few swings."""
    pts = sorted(swings, key=lambda s: s.index)
    if len(pts) < 3:
        return MicrostructureFeatures()
    legs = np.array([pts[i + 1].price - pts[i].price for i in range(len(pts) - 1)])
    times = np.array([pts[i + 1].index - pts[i].index for i in range(len(pts) - 1)], dtype=float)
    abs_legs = np.abs(legs)

    up = int(trend.direction) >= 0
    with_trend = legs > 0 if up else legs < 0
    impulse_legs = abs_legs[with_trend]
    correction_legs = abs_legs[~with_trend]
    impulse_len = float(impulse_legs[-3:].mean()) if impulse_legs.size else 0.0
    correction_len = float(correction_legs[-3:].mean()) if correction_legs.size else 0.0

    last_leg = abs_legs[-1]
    prev_leg = abs_legs[-2]
    retracement = float(last_leg / prev_leg * 100.0) if prev_leg else 0.0
    velocity = float(abs_legs[-1] / times[-1]) if times[-1] else 0.0
    prev_velocity = float(abs_legs[-2] / times[-2]) if times[-2] else 0.0
    return MicrostructureFeatures(
        impulse_length=impulse_len,
        correction_length=correction_len,
        impulse_ratio=impulse_len / correction_len if correction_len else 0.0,
        retracement_pct=min(retracement, 100.0),
        extension_pct=max(retracement - 100.0, 0.0),
        swing_velocity=velocity,
        swing_acceleration=velocity - prev_velocity,
        time_between_swings=float(times[-3:].mean()),
        average_swing_length=float(abs_legs.mean()),
        valid=1.0,
    )


def build_session(df: pd.DataFrame, config: EngineConfig) -> SessionFeatures:
    """Compute session/calendar features from the last timestamp (UTC hours)."""
    ts: pd.Timestamp = pd.Timestamp(df["timestamp"].iloc[-1])
    hour = ts.hour

    def in_session(bounds: tuple[int, int]) -> bool:
        start, end = bounds
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end  # wraps midnight

    sydney = in_session(config.session_sydney)
    asian = in_session(config.session_asian)
    london = in_session(config.session_london)
    ny = in_session(config.session_newyork)
    return SessionFeatures(
        is_sydney=float(sydney),
        is_asian=float(asian),
        is_london=float(london),
        is_newyork=float(ny),
        hour=float(hour),
        minute=float(ts.minute),
        day_of_week=float(ts.dayofweek),
        month=float(ts.month),
    )


def build_structure(
    n: int, breaks: List[StructureBreak], chochs: List[ChochEvent]
) -> StructureFeatures:
    """Summarize BOS/CHOCH history into last-event scalars and counts."""
    feat = StructureFeatures(
        bullish_bos_count=float(sum(b.direction == "bullish" for b in breaks)),
        bearish_bos_count=float(sum(b.direction == "bearish" for b in breaks)),
        choch_count=float(len(chochs)),
    )
    if breaks:
        last = breaks[-1]
        feat.last_bos_direction = 1.0 if last.direction == "bullish" else -1.0
        feat.last_bos_strength = last.strength
        feat.bars_since_bos = float(n - 1 - last.index)
    if chochs:
        last_c = chochs[-1]
        feat.last_choch_direction = 1.0 if last_c.direction == "bullish" else -1.0
        feat.last_choch_strength = last_c.strength
        feat.bars_since_choch = float(n - 1 - last_c.index)
    return feat
