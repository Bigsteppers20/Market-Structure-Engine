"""Shared, dependency-free numeric utilities.

Small vectorized helpers reused by several engines. No module-level state.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Vectorized True Range.

    ``TR[i] = max(high-low, |high-prev_close|, |low-prev_close|)`` with the
    first element falling back to ``high-low``.
    """
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    return np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )


def wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (RMA), equivalent to ``ewm(alpha=1/period)``."""
    series = pd.Series(values)
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().to_numpy()


def safe_divide(
    numerator: np.ndarray, denominator: np.ndarray, fill: float = 0.0
) -> np.ndarray:
    """Elementwise division that returns ``fill`` where the denominator is 0/NaN."""
    denominator = np.asarray(denominator, dtype=float)
    numerator = np.asarray(numerator, dtype=float)
    out = np.full_like(numerator, fill, dtype=float)
    mask = np.isfinite(denominator) & (denominator != 0.0)
    out[mask] = numerator[mask] / denominator[mask]
    return out


def last_valid(values: np.ndarray, default: float = 0.0) -> float:
    """Last finite value of an array, or ``default`` if none exists."""
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return float(default)
    return float(arr[np.nonzero(mask)[0][-1]])


def scalar(value: Optional[float], default: float = 0.0) -> float:
    """Coerce an optional/NaN value to a plain finite float."""
    if value is None:
        return float(default)
    value = float(value)
    return value if np.isfinite(value) else float(default)


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling maximum with ``min_periods=1``."""
    return pd.Series(values).rolling(window, min_periods=1).max().to_numpy()


def rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling minimum with ``min_periods=1``."""
    return pd.Series(values).rolling(window, min_periods=1).min().to_numpy()
