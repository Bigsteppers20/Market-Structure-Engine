"""End-to-end Decision Engine demonstration against real OANDA historical
data: evaluate a Strategy Lab strategy, train a Linear Regression model and
a Logistic Regression model, then combine all three into one
``DecisionResult`` -- the platform's single source of truth.

Mirrors the required flow, entirely in memory:

    Live Broker Data -> Market Structure Engine -> MarketState
                      -> Strategy Engine -> StrategyEvaluation
                      -> Linear Regression Engine -> RegressionPrediction
                      -> Logistic Regression Engine -> ClassificationPrediction
                      -> Decision Engine -> DecisionResult

Run from the project root:

    .venv\\Scripts\\python.exe examples\\decision_engine_example.py
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

from strategy import StrategyEngine
from strategies.trend_following import TrendFollowingStrategy, default_config as trend_default_config

from linear_regression import RegressionConfig, RegressionEngine
from logistic_regression import ClassificationConfig, LogisticRegressionEngine
from logistic_regression.label_manager import ConfigurableClassificationLabelGenerator

from decision_engine import DecisionEngine, DecisionEngineConfig
from decision_engine.validator import validate_decision_result_dict

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "decision_output"
SYMBOL, TIMEFRAME = "EUR_USD", "M5"
WINDOW_SIZE, HORIZON, STRIDE = 200, 5, 3


def main() -> None:
    print(f"OANDA environment : {'PRACTICE' if PRACTICE else 'LIVE'} ({BASE_URL})")
    candles = fetch_candles(SYMBOL, TIMEFRAME, count=2500)
    print(f"Fetched {len(candles)} live candles ({candles['timestamp'].iloc[0]} -> {candles['timestamp'].iloc[-1]})")

    # ------------------------------------------------------------------ #
    # Linear Regression
    # ------------------------------------------------------------------ #
    print("\nTraining Linear Regression (next_close, expected_pip_movement)...")
    reg_ml_cfg = MLDatasetConfig(
        window_size=WINDOW_SIZE, horizon=HORIZON, stride=STRIDE, symbol=SYMBOL, timeframe=TIMEFRAME,
        regression_targets=["next_close", "expected_pip_movement"],
    )
    reg_dataset = DatasetBuilder(reg_ml_cfg).build(candles)
    reg_split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(reg_dataset))[0]
    reg_training_config = TrainingConfig(
        experiment_name="de_example_lr", output_root=str(OUTPUT_ROOT / "lr"), scaler="standard", random_seed=42,
    )
    reg_config = RegressionConfig(
        targets=["next_close", "expected_pip_movement"], prediction_horizon=HORIZON, model_type="ridge",
        n_bootstrap=10, symbol=SYMBOL, timeframe=TIMEFRAME, training_config=reg_training_config,
    )
    reg_engine = RegressionEngine(reg_config)
    reg_records = reg_engine.train(reg_dataset, reg_split)
    reg_engine.load_from_records(reg_records)
    print(f"  Test R2 (next_close): {reg_records['next_close'].testing_metrics['r2']:.3f}")

    # ------------------------------------------------------------------ #
    # Logistic Regression
    # ------------------------------------------------------------------ #
    print("\nTraining Logistic Regression (SELL/NO_TRADE/BUY)...")
    cls_ml_cfg = MLDatasetConfig(window_size=WINDOW_SIZE, horizon=HORIZON, stride=STRIDE, symbol=SYMBOL, timeframe=TIMEFRAME)
    label_generator = ConfigurableClassificationLabelGenerator(min_pip_movement=5.0, risk_reward_threshold=1.5)
    cls_dataset = DatasetBuilder(cls_ml_cfg, label_generator=label_generator).build(candles)
    cls_split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(cls_dataset))[0]
    cls_training_config = TrainingConfig(
        experiment_name="de_example_lgr", output_root=str(OUTPUT_ROOT / "lgr"), scaler="standard", random_seed=42,
    )
    cls_config = ClassificationConfig(
        prediction_horizon=HORIZON, min_pip_movement=5.0, risk_reward_threshold=1.5,
        class_balancing="class_weight", calibration_method="platt", n_bootstrap=10,
        symbol=SYMBOL, timeframe=TIMEFRAME, training_config=cls_training_config,
    )
    cls_engine = LogisticRegressionEngine(cls_config)
    cls_record = cls_engine.train(cls_dataset, cls_split)
    cls_engine.load_from_record(cls_record)
    print(f"  Test accuracy: {cls_record.testing_metrics['accuracy']:.3f}")

    # ------------------------------------------------------------------ #
    # Strategy Engine
    # ------------------------------------------------------------------ #
    strategy_engine = StrategyEngine()
    strategy_engine.register_strategy(TrendFollowingStrategy(trend_default_config()))

    # ------------------------------------------------------------------ #
    # Live decision, entirely in memory
    # ------------------------------------------------------------------ #
    mse_engine = MarketStructureEngine(EngineConfig(swing_window=5))
    mse_engine.load(candles.iloc[-WINDOW_SIZE:])
    mse_engine.analyze()
    market_state = mse_engine.market_state()
    print(f"\nLive MarketState: n_candles={market_state.n_candles}, trend={market_state.trend.direction.name}")

    strategy_eval = strategy_engine.evaluate(market_state, "trend_following", symbol=SYMBOL, timeframe=TIMEFRAME)
    regression_pred = reg_engine.predict(market_state, symbol=SYMBOL, timeframe=TIMEFRAME)
    classification_pred = cls_engine.predict(market_state, symbol=SYMBOL, timeframe=TIMEFRAME)

    print(f"Strategy recommendation: {strategy_eval.recommendation} (bias={strategy_eval.market_bias}, "
          f"compliance={strategy_eval.strategy_compliance:.1f}%, confidence={strategy_eval.strategy_confidence:.1f}%)")
    print(f"Linear Regression: expected_pip_move={regression_pred.expected_pip_move:.1f}, "
          f"confidence={regression_pred.prediction_confidence:.1f}%")
    print(f"Logistic Regression: predicted_class={classification_pred.predicted_class}, "
          f"confidence={classification_pred.prediction_confidence:.1f}%")

    decision_engine = DecisionEngine(DecisionEngineConfig())
    decision = decision_engine.decide(
        market_state=market_state, strategy_evaluation=strategy_eval,
        regression_prediction=regression_pred, classification_prediction=classification_pred,
        symbol=SYMBOL, timeframe=TIMEFRAME,
    )

    result_dict = decision.to_dict()
    issues = validate_decision_result_dict(result_dict)
    print(f"\nJSON schema validation issues: {issues or 'none'}")

    print("\n=== DecisionResult ===")
    print(json.dumps(result_dict, indent=2, default=str))

    print("\nExplanation summary:")
    print(f"  {decision.reasoning.summary}")
    for factor in decision.reasoning.supporting_factors:
        print(f"  + {factor}")
    for factor in decision.reasoning.opposing_factors:
        print(f"  - {factor}")


if __name__ == "__main__":
    main()
