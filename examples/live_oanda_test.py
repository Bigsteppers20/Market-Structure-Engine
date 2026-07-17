"""Live OANDA data test cases for the Market Structure Engine.

Fetches REAL candles from the OANDA v20 REST API (practice or live account,
per OANDA_PRACTICE in .env) -- no synthetic/mock data -- and runs them
through the exact same pipeline as examples/basic_usage.py so the output
format/shape can be compared directly against that reference run.

Credentials are read from environment variables (see market_structure/.env):
    OANDA_API_KEY
    OANDA_ACCOUNT_ID
    OANDA_PRACTICE   ("True" / "False")

Run from the project root:

    .venv\\Scripts\\python.exe examples\\live_oanda_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from market_structure import EngineConfig, MarketStructureEngine
from oanda_client import BASE_URL, PRACTICE, fetch_candles


def run_case(label: str, instrument: str, granularity: str, count: int) -> None:
    print("=" * 70)
    print(f"{label}: {instrument} {granularity} (live OANDA candles, count={count})")
    print("=" * 70)

    data = fetch_candles(instrument, granularity, count)
    print(f"Fetched           : {len(data)} live candles "
          f"({data['timestamp'].iloc[0]} -> {data['timestamp'].iloc[-1]})")

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
    print()


def main() -> None:
    print(f"OANDA environment : {'PRACTICE' if PRACTICE else 'LIVE'} ({BASE_URL})")
    print()
    run_case("Test case 1", "EUR_USD", "M5", 2000)
    run_case("Test case 2", "EUR_USD", "H1", 2000)


if __name__ == "__main__":
    sys.exit(main())
