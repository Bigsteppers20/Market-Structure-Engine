"""Live Strategy Engine demonstration against real OANDA historical data.

Mirrors the exact live-operation flow described in the Strategy Engine
spec, entirely in memory, no CSV/offline step:

    MarketStructureEngine.analyze() -> MarketState -> StrategyEngine.evaluate() -> StrategyEvaluation

Run from the project root:

    .venv\\Scripts\\python.exe examples\\strategy_evaluation_example.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market_structure import EngineConfig, MarketStructureEngine
from oanda_client import BASE_URL, PRACTICE, fetch_candles

from strategy import StrategyEngine, default_registry
from strategies.ict_strategy import IctStrategy, default_config as ict_config
from strategies.london_breakout import LondonBreakoutStrategy, default_config as london_config
from strategies.scalping_strategy import ScalpingStrategy, default_config as scalping_config
from strategies.swing_strategy import SwingStrategy, default_config as swing_config
from strategies.trend_following import TrendFollowingStrategy, default_config as trend_config


def main() -> None:
    print(f"OANDA environment : {'PRACTICE' if PRACTICE else 'LIVE'} ({BASE_URL})")
    symbol, timeframe = "EUR_USD", "M5"
    candles = fetch_candles(symbol, timeframe, count=2000)
    print(f"Fetched {len(candles)} live candles ({candles['timestamp'].iloc[0]} -> {candles['timestamp'].iloc[-1]})")

    # Market Structure Engine (unmodified, public API only) -> MarketState
    mse_engine = MarketStructureEngine(EngineConfig(swing_window=5))
    mse_engine.load(candles)
    mse_engine.analyze()
    market_state = mse_engine.market_state()
    print(f"MarketState built: n_candles={market_state.n_candles}, "
          f"trend={market_state.trend.direction.name}")

    # Strategy Engine: register every built-in strategy with its default config
    engine = StrategyEngine(default_registry())
    engine.register_strategy(IctStrategy(ict_config()))
    engine.register_strategy(TrendFollowingStrategy(trend_config()))
    engine.register_strategy(LondonBreakoutStrategy(london_config()))
    engine.register_strategy(SwingStrategy(swing_config()))
    engine.register_strategy(ScalpingStrategy(scalping_config()))

    print()
    print(f"{'Strategy':<16} {'Bias':<15} {'Recommendation':<14} {'Compliance':>10} {'Confidence':>10}")
    print("-" * 70)
    evaluations = engine.evaluate_all(market_state, symbol=symbol, timeframe=timeframe)
    for name, ev in evaluations.items():
        print(f"{name:<16} {ev.market_bias:<15} {ev.recommendation:<14} "
              f"{ev.strategy_compliance:>9.1f}% {ev.strategy_confidence:>9.1f}%")

    print()
    print("=== Full ICT evaluation ===")
    ict_eval = evaluations["ict"]
    print(json.dumps(ict_eval.to_dict(), indent=2, default=str))

    print()
    print("Explanations:")
    for line in ict_eval.explanations:
        print(f"  {line}")


if __name__ == "__main__":
    main()
