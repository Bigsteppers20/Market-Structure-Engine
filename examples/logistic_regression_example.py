"""End-to-end Logistic Regression Engine demonstration against real OANDA
historical data: train a SELL/NO_TRADE/BUY classifier (Platt-calibrated,
class-weight balanced), then predict live from a real MarketState.

Mirrors the required live-operation flow, entirely in memory:

    Live Broker Data -> Market Structure Engine -> MarketState
                      -> Logistic Regression Engine -> ClassificationPrediction

Run from the project root:

    .venv\\Scripts\\python.exe examples\\logistic_regression_example.py
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

from logistic_regression import ClassificationConfig, LogisticRegressionEngine
from logistic_regression.label_manager import ConfigurableClassificationLabelGenerator

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "lgr_output"


def main() -> None:
    print(f"OANDA environment : {'PRACTICE' if PRACTICE else 'LIVE'} ({BASE_URL})")
    symbol, timeframe = "EUR_USD", "M5"
    candles = fetch_candles(symbol, timeframe, count=2500)
    print(f"Fetched {len(candles)} live candles ({candles['timestamp'].iloc[0]} -> {candles['timestamp'].iloc[-1]})")

    window_size, horizon, stride = 250, 5, 3
    ml_cfg = MLDatasetConfig(window_size=window_size, horizon=horizon, stride=stride, symbol=symbol, timeframe=timeframe)

    # ConfigurableClassificationLabelGenerator plugs into the Dataset
    # Builder's own pluggable label_generator interface -- no duplicated
    # label-generation logic (per the spec's explicit instruction to reuse
    # the existing infrastructure, not reimplement it).
    label_generator = ConfigurableClassificationLabelGenerator(
        min_pip_movement=5.0, risk_reward_threshold=1.5,
    )
    print(f"\nBuilding dataset: window_size={window_size}, horizon={horizon}, stride={stride}...")
    dataset = DatasetBuilder(ml_cfg, label_generator=label_generator).build(candles)
    print(f"Built {len(dataset)} samples x {dataset.X.shape[1]} features")
    print(f"Classes: {dataset.class_names}")
    import numpy as np
    values, counts = np.unique(dataset.y_cls, return_counts=True)
    print(f"Class distribution: {dict(zip((dataset.class_names[v] for v in values), counts.tolist()))}")

    split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(dataset))[0]
    print(f"Split -> train={len(split.train_idx)} val={len(split.val_idx)} test={len(split.test_idx)}")

    training_config = TrainingConfig(
        experiment_name="lgr_eurusd_m5", feature_version="1.0.0", dataset_version="1.0.0",
        strategy_version="1.0.0", output_root=str(OUTPUT_ROOT), scaler="standard", random_seed=42,
    )
    classification_config = ClassificationConfig(
        prediction_horizon=horizon, min_pip_movement=5.0, risk_reward_threshold=1.5,
        class_balancing="class_weight", calibration_method="platt", threshold_strategy="argmax",
        n_bootstrap=15, pip_size=0.0001, symbol=symbol, timeframe=timeframe,
        training_config=training_config,
    )

    engine = LogisticRegressionEngine(classification_config)
    print("\nTraining SELL/NO_TRADE/BUY classifier "
          "(class_weight balancing, Platt calibration, 15-model bootstrap ensemble)...")
    record = engine.train(dataset, split)
    m = record.testing_metrics
    print(f"\nTest accuracy={m['accuracy']:.3f}  balanced_accuracy={m['balanced_accuracy']:.3f}  "
          f"f1={m['f1']:.3f}  roc_auc={m['roc_auc']:.3f}  log_loss={m['log_loss']:.3f}  "
          f"brier_score={m['brier_score']:.3f}")

    print(f"\nRegistered models: {engine.registry.list_models()}")

    # --- Load for inference ------------------------------------------------ #
    engine.load_from_record(record)

    # --- Live prediction from a REAL MarketState ---------------------------- #
    mse_engine = MarketStructureEngine(EngineConfig(swing_window=5))
    mse_engine.load(candles.iloc[-window_size:])
    mse_engine.analyze()
    market_state = mse_engine.market_state()
    print(f"\nLive MarketState: n_candles={market_state.n_candles}, trend={market_state.trend.direction.name}")

    prediction = engine.predict(market_state, symbol=symbol, timeframe=timeframe)
    print("\n=== ClassificationPrediction ===")
    print(json.dumps(prediction.to_dict(), indent=2, default=str))

    print("\nExplanation:")
    for line in prediction.explanation:
        print(f"  {line}")


if __name__ == "__main__":
    main()
