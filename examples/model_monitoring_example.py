"""End-to-end Model Monitoring demonstration against real OANDA historical
data: train a Linear Regression model and a Logistic Regression model,
attach a ``ModelMonitor`` to each, replay recent live predictions, resolve
their outcomes, evaluate health/drift, get a retraining recommendation, and
produce the structured report the Agentic AI would consume.

Mirrors the required flow, entirely in memory:

    Live Broker Data -> Market Structure Engine -> MarketState
                      -> Linear/Logistic Regression Engine -> Prediction
                      -> ModelMonitor -> health / drift / retraining report

Run from the project root:

    .venv\\Scripts\\python.exe examples\\model_monitoring_example.py
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market_structure import EngineConfig, MarketStructureEngine
from ml_pipeline import DatasetBuilder, TimeSeriesSplitter
from ml_pipeline.config import DatasetConfig as MLDatasetConfig
from ml_pipeline.label_generator import REGRESSION_REGISTRY
from oanda_client import BASE_URL, PRACTICE, fetch_candles
from training.config import TrainingConfig

from linear_regression import RegressionConfig, RegressionEngine
from logistic_regression import ClassificationConfig, LogisticRegressionEngine
from logistic_regression.label_manager import ConfigurableClassificationLabelGenerator
from model_monitor import ModelLifecycleMetadata, ModelMonitor, MonitorConfig

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "monitor_output"
SYMBOL, TIMEFRAME = "EUR_USD", "M5"
WINDOW_SIZE, HORIZON, STRIDE = 200, 5, 3
N_LIVE_REPLAY = 60


def _build_market_state(df, start, window_size):
    engine = MarketStructureEngine(EngineConfig(swing_window=5))
    engine.load(df.iloc[start: start + window_size])
    engine.analyze()
    return engine.market_state()


def _replay_regression(engine, monitor, df, window_size, n_live):
    start = len(df) - window_size - HORIZON - n_live
    for i in range(n_live):
        decision_idx = start + i + window_size - 1
        market_state = _build_market_state(df, start + i, window_size)
        prediction = engine.predict(market_state, symbol=SYMBOL, timeframe=TIMEFRAME)
        vec, names = market_state.to_vector()
        snapshot = ModelMonitor.from_regression_prediction(
            prediction, model_name=monitor.model_name, primary_target="next_close",
            decision_index=decision_idx, feature_vector=vec, feature_names=names,
        )
        monitor.record_prediction(snapshot)


def _replay_classification(engine, monitor, df, window_size, n_live):
    start = len(df) - window_size - HORIZON - n_live
    for i in range(n_live):
        decision_idx = start + i + window_size - 1
        market_state = _build_market_state(df, start + i, window_size)
        prediction = engine.predict(market_state, symbol=SYMBOL, timeframe=TIMEFRAME)
        vec, names = market_state.to_vector()
        snapshot = ModelMonitor.from_classification_prediction(
            prediction, model_name=monitor.model_name, decision_index=decision_idx,
            feature_vector=vec, feature_names=names,
        )
        monitor.record_prediction(snapshot)


def main() -> None:
    print(f"OANDA environment : {'PRACTICE' if PRACTICE else 'LIVE'} ({BASE_URL})")
    candles = fetch_candles(SYMBOL, TIMEFRAME, count=2500)
    print(f"Fetched {len(candles)} live candles ({candles['timestamp'].iloc[0]} -> {candles['timestamp'].iloc[-1]})")

    monitor_config = MonitorConfig(
        health_threshold=65.0, feature_drift_threshold=0.3, min_new_samples=20,
        rolling_window=25, market_regime_lookback=40, max_model_age_days=9999,
        retraining_mode="manual",
    )

    # ------------------------------------------------------------------ #
    # Linear Regression: train, monitor, replay live predictions
    # ------------------------------------------------------------------ #
    print("\n=== Linear Regression ===")
    reg_ml_cfg = MLDatasetConfig(
        window_size=WINDOW_SIZE, horizon=HORIZON, stride=STRIDE, symbol=SYMBOL, timeframe=TIMEFRAME,
        regression_targets=["next_close"],
    )
    reg_dataset = DatasetBuilder(reg_ml_cfg).build(candles)
    reg_split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(reg_dataset))[0]

    reg_training_config = TrainingConfig(
        experiment_name="monitor_lr_eurusd", feature_version="1.0.0", dataset_version="1.0.0",
        strategy_version="1.0.0", output_root=str(OUTPUT_ROOT / "lr"), scaler="standard", random_seed=42,
    )
    reg_config = RegressionConfig(
        targets=["next_close"], prediction_horizon=HORIZON, model_type="ridge",
        model_hyperparameters={"alpha": 1.0}, n_bootstrap=10, symbol=SYMBOL, timeframe=TIMEFRAME,
        training_config=reg_training_config,
    )
    reg_engine = RegressionEngine(reg_config)
    reg_records = reg_engine.train(reg_dataset, reg_split)
    reg_engine.load_from_records(reg_records)
    print(f"Trained + loaded. Test R2: {reg_records['next_close'].testing_metrics['r2']:.3f}")

    reg_monitor = ModelMonitor(
        monitor_config, model_name="lr_eurusd_next_close", task_type="regression", output_root=OUTPUT_ROOT / "monitor",
    )
    reg_record = reg_records["next_close"]
    reg_monitor.register_model(ModelLifecycleMetadata(
        model_name="lr_eurusd_next_close", version=reg_record.experiment_id, task_type="regression",
        status="candidate", training_date=reg_record.timestamp, training_dataset_size=len(reg_dataset),
        feature_version="1.0.0", training_version="1.0.0", strategy_version="1.0.0", dataset_version="1.0.0",
        supported_symbols=[SYMBOL], supported_timeframes=[TIMEFRAME],
    ), as_production=True)

    X_train = reg_dataset.X[reg_split.train_idx]
    reg_monitor.fit_baseline(
        X_train, reg_dataset.feature_names, target_values=reg_dataset.y_reg["next_close"][reg_split.train_idx],
    )
    _replay_regression(reg_engine, reg_monitor, candles, WINDOW_SIZE, N_LIVE_REPLAY)
    resolver = functools.partial(REGRESSION_REGISTRY["next_close"], pip_size=0.0001)
    resolved = reg_monitor.resolve_outcomes(candles, resolver)
    print(f"Logged {len(reg_monitor.prediction_log)} live predictions, resolved {len(resolved)} "
          f"(coverage={reg_monitor.prediction_log.coverage():.2f})")

    reg_health = reg_monitor.evaluate_health()
    print(f"Health: {reg_health.health.overall:.1f} ({reg_health.status})")
    reg_rec = reg_monitor.recommend_retraining(current_horizon=HORIZON)
    print(f"Retraining recommendation: recommended={reg_rec.recommended}, priority={reg_rec.priority}")
    for reason in reg_rec.reasons:
        print(f"  - {reason}")
    print("Agentic report:")
    print(json.dumps(reg_monitor.to_agentic_report(model_display_name="Linear Regression"), indent=2))

    # ------------------------------------------------------------------ #
    # Logistic Regression: train, monitor, replay live predictions
    # ------------------------------------------------------------------ #
    print("\n=== Logistic Regression ===")
    cls_ml_cfg = MLDatasetConfig(window_size=WINDOW_SIZE, horizon=HORIZON, stride=STRIDE, symbol=SYMBOL, timeframe=TIMEFRAME)
    label_generator = ConfigurableClassificationLabelGenerator(min_pip_movement=5.0, risk_reward_threshold=1.5)
    cls_dataset = DatasetBuilder(cls_ml_cfg, label_generator=label_generator).build(candles)
    cls_split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(cls_dataset))[0]

    cls_training_config = TrainingConfig(
        experiment_name="monitor_lgr_eurusd", feature_version="1.0.0", dataset_version="1.0.0",
        strategy_version="1.0.0", output_root=str(OUTPUT_ROOT / "lgr"), scaler="standard", random_seed=42,
    )
    cls_config = ClassificationConfig(
        prediction_horizon=HORIZON, min_pip_movement=5.0, risk_reward_threshold=1.5,
        class_balancing="class_weight", calibration_method="platt", n_bootstrap=10,
        symbol=SYMBOL, timeframe=TIMEFRAME, training_config=cls_training_config,
    )
    cls_engine = LogisticRegressionEngine(cls_config)
    cls_record = cls_engine.train(cls_dataset, cls_split)
    cls_engine.load_from_record(cls_record)
    print(f"Trained + loaded. Test accuracy: {cls_record.testing_metrics['accuracy']:.3f}")

    cls_monitor = ModelMonitor(
        monitor_config, model_name="lgr_eurusd", task_type="classification",
        output_root=OUTPUT_ROOT / "monitor", classes=["SELL", "NO_TRADE", "BUY"],
    )
    cls_monitor.register_model(ModelLifecycleMetadata(
        model_name="lgr_eurusd", version=cls_record.experiment_id, task_type="classification",
        status="candidate", training_date=cls_record.timestamp, training_dataset_size=len(cls_dataset),
        feature_version="1.0.0", training_version="1.0.0", strategy_version="1.0.0", dataset_version="1.0.0",
        supported_symbols=[SYMBOL], supported_timeframes=[TIMEFRAME],
    ), as_production=True)

    X_train_cls = cls_dataset.X[cls_split.train_idx]
    cls_monitor.fit_baseline(X_train_cls, cls_dataset.feature_names)
    _replay_classification(cls_engine, cls_monitor, candles, WINDOW_SIZE, N_LIVE_REPLAY)
    resolved_cls = cls_monitor.resolve_outcomes(candles, label_generator.label)
    print(f"Logged {len(cls_monitor.prediction_log)} live predictions, resolved {len(resolved_cls)} "
          f"(coverage={cls_monitor.prediction_log.coverage():.2f})")

    cls_health = cls_monitor.evaluate_health()
    print(f"Health: {cls_health.health.overall:.1f} ({cls_health.status})")
    cls_rec = cls_monitor.recommend_retraining(current_horizon=HORIZON)
    print(f"Retraining recommendation: recommended={cls_rec.recommended}, priority={cls_rec.priority}")
    print("Agentic report:")
    print(json.dumps(cls_monitor.to_agentic_report(model_display_name="Logistic Regression"), indent=2))

    # ------------------------------------------------------------------ #
    # Adaptive retraining, simulated: force an aggressive policy so the
    # AND-gate actually fires, then run one retraining cycle end-to-end
    # (train candidate -> compare -> promote/archive), never touching the
    # engines' training code directly -- model_monitor only calls whatever
    # zero-arg callback the caller supplies.
    # ------------------------------------------------------------------ #
    print("\n=== Simulated adaptive retraining cycle (Linear Regression) ===")
    from model_monitor import CandidateArtifact
    from model_monitor.config import MonitorConfig as _MonitorConfig

    aggressive_config = _MonitorConfig(
        health_threshold=99.9, feature_drift_threshold=0.0, min_new_samples=1,
        auto_retraining_enabled=True, retraining_mode="adaptive", max_model_age_days=9999,
    )
    reg_monitor.config = aggressive_config
    reg_monitor.retraining_manager.config = aggressive_config

    def train_candidate() -> CandidateArtifact:
        print("  (train_candidate_fn invoked -- in production this would call RegressionEngine.train() again)")
        prod_metrics = reg_health.performance.historical.metrics
        return CandidateArtifact(
            version="candidate_v2", metrics={"rmse": prod_metrics["rmse"] * 0.7},
            training_date=reg_health.evaluated_at, training_dataset_size=len(reg_dataset) + 500,
        )

    outcome = reg_monitor.check_retraining(current_horizon=HORIZON, train_candidate_fn=train_candidate)
    print(f"Retraining outcome: triggered={outcome.triggered}, reason={outcome.reason}")
    if outcome.comparison:
        print(f"  Comparison decision: {outcome.comparison.decision} "
              f"({outcome.comparison.primary_metric}: {outcome.comparison.production_value:.6f} -> "
              f"{outcome.comparison.candidate_value:.6f})")
    print(f"  Promoted version: {outcome.promoted_version}")
    print(f"  Production is now: {reg_monitor.lifecycle_registry.production_version('lr_eurusd_next_close').version}")

    print("\nNotifications emitted:")
    for note in reg_monitor.notifications.history:
        print(f"  [{note.severity}] {note.type}: {note.message}")


if __name__ == "__main__":
    main()
