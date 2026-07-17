"""Configuration for the Market Structure Engine.

All tunable parameters live in :class:`EngineConfig` so that every analysis
module is deterministic and independently testable with explicit settings.
No module reads global state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(slots=True)
class EngineConfig:
    """Tunable parameters for every sub-engine.

    Attributes are grouped by the module that consumes them. Defaults are
    sensible for intraday forex data (M5-H1) but everything is overridable.
    """

    # --- swings -----------------------------------------------------------
    swing_window: int = 5
    """Bars on each side a pivot must dominate to qualify as a swing."""

    # --- trend ------------------------------------------------------------
    trend_swing_lookback: int = 6
    """Number of most recent swings used to classify the trend."""
    sideways_threshold: float = 0.25
    """|net directional score| below this fraction => sideways market."""

    # --- support / resistance ----------------------------------------------
    zone_merge_atr_multiple: float = 0.5
    """Swing levels closer than this many ATRs are merged into one zone."""
    max_zones: int = 10
    """Maximum number of zones kept per side (strongest first)."""

    # --- liquidity ----------------------------------------------------------
    equal_level_atr_tolerance: float = 0.15
    """Highs/lows within this many ATRs count as 'equal' (a pool)."""
    sweep_close_back_ratio: float = 0.5
    """Fraction of the sweep wick that must be given back for a sweep."""

    # --- fair value gaps ----------------------------------------------------
    fvg_min_atr_multiple: float = 0.1
    """Gaps smaller than this many ATRs are ignored as noise."""

    # --- order blocks -------------------------------------------------------
    ob_displacement_atr_multiple: float = 1.5
    """Impulse after the OB candle must exceed this many ATRs."""
    ob_max_age: int = 500
    """Order blocks older than this many bars are dropped."""

    # --- indicators ---------------------------------------------------------
    ema_periods: Tuple[int, ...] = (20, 50, 100, 200)
    sma_period: int = 20
    atr_period: int = 14
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14
    momentum_period: int = 10
    roc_period: int = 10
    cci_period: int = 20
    stoch_k: int = 14
    stoch_d: int = 3
    williams_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    rolling_window: int = 20
    """Window for rolling mean / median / variance / volatility."""

    # --- volume -------------------------------------------------------------
    volume_ma_period: int = 20
    volume_spike_multiple: float = 2.0
    """Relative volume above this multiple flags a volume spike."""

    # --- volatility ---------------------------------------------------------
    hist_vol_window: int = 20
    expansion_ratio: float = 1.5
    """Current ATR / rolling ATR mean above this => expansion regime."""
    compression_ratio: float = 0.66
    """Current ATR / rolling ATR mean below this => compression regime."""

    # --- spread ---------------------------------------------------------
    spread_window: int = 20
    """Rolling window for spread average / volatility / percentile features."""
    spread_spike_multiple: float = 2.0
    """Spread above this multiple of its rolling average flags a spread spike."""

    # --- data loader ---------------------------------------------------------
    fill_missing_candles: bool = False
    """When True, reindex to a regular grid and forward-fill gaps."""

    # --- sessions (UTC hours, inclusive start / exclusive end) ---------------
    session_sydney: Tuple[int, int] = (21, 6)
    session_asian: Tuple[int, int] = (0, 9)
    session_london: Tuple[int, int] = (7, 16)
    session_newyork: Tuple[int, int] = (12, 21)

    extra: dict = field(default_factory=dict)
    """Free-form overrides for experimentation; never read by core modules."""
