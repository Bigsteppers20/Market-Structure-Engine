"""Basic usage of the Market Structure Engine.

Generates synthetic OHLCV data, runs the full analysis, and prints the
resulting feature vector — the exact array a downstream ML model would
consume. Run from the project root:

    python examples/basic_usage.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from market_structure import EngineConfig, MarketStructureEngine


def make_synthetic_ohlcv(n: int = 2_000, seed: int = 42) -> pd.DataFrame:
    """Random-walk EURUSD-like candles with valid geometry."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.00002, 0.0008, n)
    close = 1.1000 * np.exp(np.cumsum(rets))
    open_ = np.concatenate(([1.1000], close[:-1]))
    wick_hi = np.abs(rng.normal(0, 0.0004, n))
    wick_lo = np.abs(rng.normal(0, 0.0004, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-06-02", periods=n, freq="5min"),
            "open": open_,
            "high": np.maximum(open_, close) + wick_hi,
            "low": np.minimum(open_, close) - wick_lo,
            "close": close,
            "volume": rng.integers(50, 800, n).astype(float),
        }
    )


def main() -> None:
    data = make_synthetic_ohlcv()

    engine = MarketStructureEngine(EngineConfig(swing_window=5))
    engine.load(data)
    engine.analyze()

    state = engine.market_state()
    vector, names = engine.feature_vector()

    print(f"Candles analyzed : {state.n_candles}")
    print(f"Trend            : {state.trend.direction.name} "
          f"(strength={state.trend.strength:.2f}, momentum={state.trend.momentum:+.6f})")
    print(f"Swings           : {len(state.swings)}")
    print(f"BOS / CHOCH      : {len(state.breaks)} / {len(state.chochs)}")
    print(f"Unfilled FVGs    : {state.fvg.unfilled_count}")
    print(f"Order blocks     : {state.order_blocks.unmitigated_count} unmitigated")
    print(f"Feature vector   : {vector.shape[0]} dimensions")
    print()
    print("First 15 features:")
    for name, value in list(zip(names, vector))[:15]:
        print(f"  {name:<28} {value:+.6f}")


if __name__ == "__main__":
    main()
