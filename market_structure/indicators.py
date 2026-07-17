"""Technical indicators, implemented manually with pandas/numpy vectorization.

No TA library is used. Every function returns a full ``np.ndarray`` aligned
to the input series; :class:`IndicatorEngine.analyze` assembles them into an
:class:`IndicatorPanel` of arrays plus a snapshot of last values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from .config import EngineConfig
from .utils import safe_divide, true_range, wilder_smooth


# --------------------------------------------------------------------------- #
# primitive indicator functions
# --------------------------------------------------------------------------- #
def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with span = period."""
    return pd.Series(values).ewm(span=period, adjust=False).mean().to_numpy()


def sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    return pd.Series(values).rolling(period, min_periods=1).mean().to_numpy()


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range (Wilder smoothing)."""
    return wilder_smooth(true_range(high, low, close), period)


def rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index (Wilder)."""
    delta = np.diff(close, prepend=close[0])
    gain = wilder_smooth(np.where(delta > 0, delta, 0.0), period)
    loss = wilder_smooth(np.where(delta < 0, -delta, 0.0), period)
    rs = safe_divide(gain, loss, fill=np.inf)
    out = 100.0 - 100.0 / (1.0 + rs)
    return np.where(np.isfinite(out), out, 100.0)


def macd(
    close: np.ndarray, fast: int, slow: int, signal: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, histogram."""
    line = ema(close, fast) - ema(close, slow)
    sig = pd.Series(line).ewm(span=signal, adjust=False).mean().to_numpy()
    return line, sig, line - sig


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average Directional Index (Wilder)."""
    up = np.diff(high, prepend=high[0])
    down = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_s = wilder_smooth(true_range(high, low, close), period)
    plus_di = 100.0 * safe_divide(wilder_smooth(plus_dm, period), tr_s)
    minus_di = 100.0 * safe_divide(wilder_smooth(minus_dm, period), tr_s)
    dx = 100.0 * safe_divide(np.abs(plus_di - minus_di), plus_di + minus_di)
    return wilder_smooth(dx, period)


def momentum(close: np.ndarray, period: int) -> np.ndarray:
    """Price momentum: close - close[period bars ago]."""
    out = np.zeros_like(close)
    out[period:] = close[period:] - close[:-period]
    return out


def roc(close: np.ndarray, period: int) -> np.ndarray:
    """Rate of change (percentage)."""
    out = np.zeros_like(close)
    out[period:] = safe_divide(close[period:] - close[:-period], close[:-period]) * 100.0
    return out


def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3.0
    s = pd.Series(tp)
    ma = s.rolling(period, min_periods=1).mean()
    mad = (s - ma).abs().rolling(period, min_periods=1).mean().to_numpy()
    return safe_divide(tp - ma.to_numpy(), 0.015 * mad)


def stochastic(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int, d_period: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic oscillator (%K, %D)."""
    ll = pd.Series(low).rolling(k_period, min_periods=1).min().to_numpy()
    hh = pd.Series(high).rolling(k_period, min_periods=1).max().to_numpy()
    k = 100.0 * safe_divide(close - ll, hh - ll, fill=50.0)
    d = pd.Series(k).rolling(d_period, min_periods=1).mean().to_numpy()
    return k, d


def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Williams %R."""
    hh = pd.Series(high).rolling(period, min_periods=1).max().to_numpy()
    ll = pd.Series(low).rolling(period, min_periods=1).min().to_numpy()
    return -100.0 * safe_divide(hh - close, hh - ll, fill=0.5)


def bollinger(
    close: np.ndarray, period: int, num_std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands (upper, middle, lower)."""
    s = pd.Series(close)
    mid = s.rolling(period, min_periods=1).mean().to_numpy()
    std = s.rolling(period, min_periods=1).std(ddof=0).to_numpy()
    std = np.where(np.isfinite(std), std, 0.0)
    return mid + num_std * std, mid, mid - num_std * std


def vwap(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """Cumulative Volume-Weighted Average Price across the loaded series."""
    tp = (high + low + close) / 3.0
    cum_v = np.cumsum(volume)
    cum_pv = np.cumsum(tp * volume)
    return np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-12), tp)


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class IndicatorPanel:
    """Full-length indicator arrays keyed by name, plus last-bar snapshot.

    Attributes
    ----------
    series:
        Mapping of indicator name -> full ``np.ndarray`` (aligned to candles).
    snapshot:
        Mapping of indicator name -> last finite value as ``float``.
    valid:
        Mapping of indicator name -> 1.0/0.0 flag telling whether ``snapshot``
        reflects a fully warmed-up reading (enough bars loaded for that
        indicator's period) rather than an early-history placeholder. Always
        check this before trusting a snapshot value -- a warm-up placeholder
        and a genuine zero/flat reading are otherwise indistinguishable.
    """

    series: Dict[str, np.ndarray] = field(default_factory=dict)
    snapshot: Dict[str, float] = field(default_factory=dict)
    valid: Dict[str, float] = field(default_factory=dict)


class IndicatorEngine:
    """Computes every configured indicator plus volume features in one pass."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def analyze(self, df: pd.DataFrame) -> IndicatorPanel:
        """Compute all indicator series and a last-value snapshot."""
        cfg = self.config
        h = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        v = df["volume"].to_numpy(dtype=float)

        s: Dict[str, np.ndarray] = {}
        for p in cfg.ema_periods:
            s[f"ema_{p}"] = ema(c, p)
        s["sma"] = sma(c, cfg.sma_period)
        s["true_range"] = true_range(h, lo, c)
        s["atr"] = atr(h, lo, c, cfg.atr_period)
        s["rsi"] = rsi(c, cfg.rsi_period)
        # MACD histogram is a pure linear combination (line - signal); both
        # inputs already appear in the vector, so it is not stored separately
        # (see FEATURE_OPTIMIZATION_REPORT.md, Task 2: redundant features).
        s["macd"], s["macd_signal"], _macd_hist = macd(
            c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
        )
        s["adx"] = adx(h, lo, c, cfg.adx_period)
        s["momentum"] = momentum(c, cfg.momentum_period)
        s["roc"] = roc(c, cfg.roc_period)
        s["cci"] = cci(h, lo, c, cfg.cci_period)
        s["stoch_k"], s["stoch_d"] = stochastic(h, lo, c, cfg.stoch_k, cfg.stoch_d)
        s["williams_r"] = williams_r(h, lo, c, cfg.williams_period)
        # Bollinger middle band is dropped: identical formula/period to `sma`
        # under default config, and fully reconstructable as sma +/- num_std
        # * std_dev otherwise.
        s["bb_upper"], _bb_mid, s["bb_lower"] = bollinger(
            c, cfg.bollinger_period, cfg.bollinger_std
        )
        s["vwap"] = vwap(h, lo, c, v)

        roll = pd.Series(c).rolling(cfg.rolling_window, min_periods=1)
        # `rolling_mean` is dropped: identical to `sma` under default config
        # (see FEATURE_OPTIMIZATION_REPORT.md). `rolling_median` is kept --
        # a robust-statistic, genuinely distinct from the mean.
        s["rolling_median"] = roll.median().to_numpy()
        # `rolling_variance` is dropped as a standalone feature: `std_dev` is
        # a deterministic monotonic transform (sqrt) carrying the same
        # information in more directly interpretable (price) units.
        _rolling_variance = np.nan_to_num(roll.var(ddof=0).to_numpy())
        s["std_dev"] = np.sqrt(_rolling_variance)
        returns = pd.Series(c).pct_change().fillna(0.0)
        s["volatility"] = (
            returns.rolling(cfg.rolling_window, min_periods=2).std(ddof=0).fillna(0.0).to_numpy()
        )

        # --- volume features -------------------------------------------------
        vol_ma = sma(v, cfg.volume_ma_period)
        rel_vol = safe_divide(v, vol_ma, fill=1.0)
        s["volume_ma"] = vol_ma
        s["relative_volume"] = rel_vol
        s["volume_spike"] = (rel_vol >= cfg.volume_spike_multiple).astype(float)
        vol_trend = safe_divide(
            vol_ma - np.roll(vol_ma, cfg.volume_ma_period), np.roll(vol_ma, cfg.volume_ma_period)
        )
        vol_trend[: cfg.volume_ma_period] = 0.0
        s["volume_trend"] = vol_trend
        direction = np.sign(c - df["open"].to_numpy(dtype=float))
        s["volume_delta"] = pd.Series(v * direction).rolling(
            cfg.volume_ma_period, min_periods=1
        ).sum().to_numpy()
        s["volume_ratio"] = safe_divide(v, np.roll(v, 1), fill=1.0)
        s["volume_ratio"][0] = 1.0
        has_tick_volume = "tick_volume" in df.columns
        if has_tick_volume:
            s["tick_volume"] = df["tick_volume"].to_numpy(dtype=float)
        else:
            # No real tick-volume feed: emit NaN-safe zeros but mark the
            # feature invalid rather than letting 0.0 masquerade as a real
            # reading (see FEATURE_OPTIMIZATION_REPORT.md, Task 5).
            s["tick_volume"] = np.zeros_like(v)

        snapshot = {name: _last_finite(arr) for name, arr in s.items()}
        n = len(df)
        warmup = _warmup_requirements(cfg)
        valid = {name: float(n >= warmup.get(name, 1)) for name in s}
        valid["tick_volume"] = float(has_tick_volume)
        return IndicatorPanel(series=s, snapshot=snapshot, valid=valid)


def _warmup_requirements(cfg: EngineConfig) -> Dict[str, int]:
    """Minimum bar count for each indicator's snapshot to be a statistically
    warmed-up reading rather than an early-history placeholder.

    Used to populate :attr:`IndicatorPanel.valid`. Values are documented
    approximations (e.g. ADX is conservatively given 2x its period to allow
    its internal DI smoothing to stabilize), not exact statistical proofs --
    see FEATURE_OPTIMIZATION_REPORT.md, Task 3.
    """
    warmup: Dict[str, int] = {f"ema_{p}": p for p in cfg.ema_periods}
    warmup.update(
        sma=cfg.sma_period,
        true_range=1,
        atr=cfg.atr_period,
        rsi=cfg.rsi_period,
        macd=max(cfg.macd_fast, cfg.macd_slow),
        macd_signal=max(cfg.macd_fast, cfg.macd_slow) + cfg.macd_signal,
        adx=cfg.adx_period * 2,
        momentum=cfg.momentum_period,
        roc=cfg.roc_period,
        cci=cfg.cci_period,
        stoch_k=cfg.stoch_k,
        stoch_d=cfg.stoch_k + cfg.stoch_d,
        williams_r=cfg.williams_period,
        bb_upper=cfg.bollinger_period,
        bb_lower=cfg.bollinger_period,
        vwap=1,
        rolling_median=cfg.rolling_window,
        std_dev=cfg.rolling_window,
        volatility=cfg.rolling_window + 1,
        volume_ma=cfg.volume_ma_period,
        relative_volume=cfg.volume_ma_period,
        volume_spike=cfg.volume_ma_period,
        volume_trend=cfg.volume_ma_period * 2,
        volume_delta=cfg.volume_ma_period,
        volume_ratio=2,
    )
    return warmup


def _last_finite(arr: np.ndarray) -> float:
    mask = np.isfinite(arr)
    if not mask.any():
        return 0.0
    return float(arr[np.nonzero(mask)[0][-1]])
