"""Shared fixtures for the Market Structure Engine test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    n: int = 500,
    seed: int = 7,
    start_price: float = 1.1000,
    drift: float = 0.0,
    vol: float = 0.0008,
    freq: str = "5min",
) -> pd.DataFrame:
    """Generate a random-walk OHLCV DataFrame with valid candle geometry."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0, vol * start_price, (2, n)))
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    volume = rng.integers(50, 500, n).astype(float)
    ts = pd.date_range("2025-01-06", periods=n, freq=freq)
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low,
         "close": close, "volume": volume}
    )


@pytest.fixture()
def random_df() -> pd.DataFrame:
    """500 candles of seeded random-walk data."""
    return make_ohlcv()


@pytest.fixture()
def trending_up_df() -> pd.DataFrame:
    """Uptrending series with enough noise to form HH/HL swing structure."""
    return make_ohlcv(n=400, seed=3, drift=0.0004, vol=0.0012)


@pytest.fixture()
def trending_down_df() -> pd.DataFrame:
    """Downtrending series with pullbacks (LH/LL structure)."""
    return make_ohlcv(n=400, seed=4, drift=-0.0004, vol=0.0012)
