# Market Structure Engine (MSE)

Transforms raw OHLCV forex data into a comprehensive numerical representation of the current market state. The engine emits **features only** — it never produces BUY / SELL / NO_TRADE decisions. Its output is designed to be consumed by downstream models (linear regression forecasting, logistic classification, agentic systems).

## Installation

```bash
pip install -e .          # from the project root
pip install matplotlib    # optional, only for visualization
```

Requires Python 3.11+, pandas, numpy, scipy. No TA libraries are used; every algorithm is implemented manually and vectorized.

## Quick start

```python
from market_structure import MarketStructureEngine

engine = MarketStructureEngine()
engine.load(data)                 # pandas DataFrame or List[Candle]
engine.analyze()

state = engine.market_state()     # MarketState dataclass
vector, names = engine.feature_vector()   # np.float64 array + feature names
```

Input requires columns `timestamp, open, high, low, close, volume`; `spread` and `tick_volume` are optional. The loader validates schema and candle geometry, sorts, de-duplicates, normalizes dtypes, and can optionally fill missing candles (`EngineConfig(fill_missing_candles=True)`).

## What the engine computes

**Structure** — swing highs/lows (strength, spacing), trend direction with HH/HL/LH/LL counts, strength, duration and momentum, break of structure events, change of character events.

**Zones and liquidity** — clustered support/resistance zones with touches, width, strength and ATR-normalized distances; equal highs/lows, liquidity pools, sweep detection, buy-side and sell-side liquidity totals.

**Imbalances** — fair value gaps (size, age, fill ratio, nearest unfilled distance) and displacement-validated order blocks (strength, freshness, mitigation, retests).

**Indicators** — EMA 20/50/100/200, SMA, ATR, RSI, MACD (+signal, histogram), ADX, momentum, ROC, CCI, stochastic, Williams %R, Bollinger bands, standard deviation, volatility, rolling mean/median/variance, VWAP, plus a full volume block (average, relative, spike, trend, delta, ratio, tick volume).

**Context** — 20 candlestick patterns encoded as 0/1, last-candle price action (returns, body/wick geometry, gaps, distances from EMAs/VWAP), volatility regime (expansion/compression), microstructure (impulse vs correction legs, retracement/extension, swing velocity/acceleration) and session/calendar features (Sydney/Asian/London/New York, overlap, hour, day of week, month).

`MarketState.to_dict()` flattens everything into an ordered `name -> float` map; `to_vector()` returns the aligned `(values, names)` pair. Ordering is deterministic across runs.

## Configuration

Every threshold lives in `EngineConfig` — swing window, trend lookback, zone merge tolerance, sweep close-back ratio, FVG minimum size, order-block displacement multiple, all indicator periods, session hours (UTC), and more. Pass a config to the engine constructor:

```python
from market_structure import EngineConfig, MarketStructureEngine

cfg = EngineConfig(swing_window=3, ob_displacement_atr_multiple=2.0)
engine = MarketStructureEngine(cfg)
```

## Performance

The pipeline is vectorized end to end. Benchmark on this reference environment: **100,000 candles analyzed in ~0.6 s** (requirement: under 2 s).

## Testing

```bash
python -m pytest tests -q --cov=market_structure
```

The suite covers every module (98% line coverage), including an integration test asserting the feature vector contains no trade-decision fields, determinism checks, and a marked performance benchmark (`-m performance`).

## Visualization (optional)

```python
from market_structure.visualization import plot_market_structure
plot_market_structure(engine.data, engine.market_state(), last_n=300, save_path="chart.png")
```

Renders candles with swings, BOS levels, S/R zones, unfilled FVGs and unmitigated order blocks. Requires matplotlib; the analysis pipeline never imports it.
