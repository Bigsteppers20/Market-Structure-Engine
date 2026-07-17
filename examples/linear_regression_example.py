"""End-to-end Linear Regression Engine demonstration against real OANDA
historical data: train on all 8 named regression targets, then predict live
from a real MarketState.

Mirrors the required live-operation flow, entirely in memory:

    Live Broker Data -> Market Structure Engine -> MarketState
                      -> Regression Engine -> RegressionPrediction

Run from the project root:

    .venv\\Scripts\\python.exe examples\\linear_regression_example.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market_structure import EngineConfig, MarketStructureEngine
from ml_pipeline import DatasetBuilder, TimeSeriesSplitter
from ml_pipeline.config import DatasetConfig as MLDatasetConfig
from oanda_client import BASE_URL, PRACTICE, fetch_candles
from training.config import TrainingConfig

from linear_regression import RegressionConfig, RegressionEngine

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "lr_output"

#: The 8 targets with a named slot on RegressionPrediction.
NAMED_TARGETS = [
    "next_close", "next_high", "next_low", "next_return",
    "expected_pip_movement", "future_volatility",
    "maximum_favorable_excursion", "maximum_adverse_excursion",
]


def main() -> None:
    print(f"OANDA environment : {'PRACTICE' if PRACTICE else 'LIVE'} ({BASE_URL})")
    symbol, timeframe = "EUR_USD", "M5"
    candles = fetch_candles(symbol, timeframe, count=2500)
    print(f"Fetched {len(candles)} live candles ({candles['timestamp'].iloc[0]} -> {candles['timestamp'].iloc[-1]})")

    window_size, horizon, stride = 250, 5, 3
    # ml_pipeline.DatasetBuilder only knows its own native target registry --
    # the 2 new-to-this-engine targets (MFE, MAE) are added afterward via
    # RegressionEngine.train()'s raw_df augmentation (see target_generator.py).
    from ml_pipeline.label_generator import REGRESSION_REGISTRY as NATIVE_REGISTRY
    native_targets = [t for t in NAMED_TARGETS if t in NATIVE_REGISTRY]
    ml_cfg = MLDatasetConfig(
        window_size=window_size, horizon=horizon, stride=stride, symbol=symbol, timeframe=timeframe,
        regression_targets=native_targets,
    )
    print(f"\nBuilding dataset: window_size={window_size}, horizon={horizon}, stride={stride}...")
    dataset = DatasetBuilder(ml_cfg).build(candles)
    print(f"Built {len(dataset)} samples x {dataset.X.shape[1]} features")
    print(f"Natively-built targets: {native_targets}")
    print(f"Targets added via augmentation: {[t for t in NAMED_TARGETS if t not in native_targets]}")

    split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(dataset))[0]
    print(f"Split -> train={len(split.train_idx)} val={len(split.val_idx)} test={len(split.test_idx)}")

    training_config = TrainingConfig(
        experiment_name="lr_eurusd_m5", feature_version="1.0.0", dataset_version="1.0.0",
        strategy_version="1.0.0", output_root=str(OUTPUT_ROOT), scaler="standard", random_seed=42,
    )
    regression_config = RegressionConfig(
        targets=NAMED_TARGETS, prediction_horizon=horizon, model_type="ridge",
        model_hyperparameters={"alpha": 1.0}, n_bootstrap=15, pip_size=0.0001,
        symbol=symbol, timeframe=timeframe, training_config=training_config,
    )

    engine = RegressionEngine(regression_config)
    print(f"\nTraining {len(NAMED_TARGETS)} independent linear models "
          f"(one per target, each a full training.Trainer.run() pass)...")
    records = engine.train(dataset, split, raw_df=candles)
    print(f"\n{'Target':<28} {'Test R2':>10} {'Test RMSE':>12} {'Explained Var':>14}")
    print("-" * 68)
    for target, record in records.items():
        m = record.testing_metrics
        print(f"{target:<28} {m['r2']:>10.3f} {m['rmse']:>12.6f} {m['explained_variance']:>14.3f}")

    print(f"\nRegistered models: {engine.registry.list_models()}")

    # --- Load for inference ------------------------------------------------ #
    engine.load_from_records(records)

    # --- Live prediction from a REAL MarketState ---------------------------- #
    # Uses the same trailing window_size the models were trained on -- see
    # LINEAR_REGRESSION_ENGINE_REPORT.md for why this matters (the confidence
    # engine's distribution-distance factor compares against the training
    # feature distribution, which was built entirely from window_size-candle
    # MarketStates).
    mse_engine = MarketStructureEngine(EngineConfig(swing_window=5))
    mse_engine.load(candles.iloc[-window_size:])
    mse_engine.analyze()
    market_state = mse_engine.market_state()
    print(f"\nLive MarketState: n_candles={market_state.n_candles}, trend={market_state.trend.direction.name}")

    prediction = engine.predict(market_state, symbol=symbol, timeframe=timeframe)
    print("\n=== RegressionPrediction ===")
    print(json.dumps(prediction.to_dict(), indent=2, default=str))

    print("\nExplanation:")
    for line in prediction.explanation:
        print(f"  {line}")


if __name__ == "__main__":
    main()
