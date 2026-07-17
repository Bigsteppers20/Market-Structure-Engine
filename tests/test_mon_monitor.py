"""End-to-end tests for model_monitor.ModelMonitor -- the full
Live Broker Data -> MarketState -> (Linear/Logistic Regression Engine) ->
ModelMonitor -> health/drift/retraining/agentic-report pipeline, against
REAL trained engines (not mocks), for both task types."""
from __future__ import annotations

import functools

import numpy as np
import pytest

from conftest import make_ohlcv
from market_structure import EngineConfig, MarketStructureEngine
from ml_pipeline.label_generator import REGRESSION_REGISTRY
from model_monitor import ModelMonitor, ModelLifecycleMetadata, MonitorConfig
from model_monitor.exceptions import InsufficientDataError, ModelMonitorError


def _live_predictions_regression(engine, df, window_size, n_live, monitor):
    start = len(df) - window_size - 5 - n_live
    for i in range(n_live):
        decision_idx = start + i + window_size - 1
        window = df.iloc[start + i: start + i + window_size]
        mse_engine = MarketStructureEngine(EngineConfig(swing_window=5))
        mse_engine.load(window)
        mse_engine.analyze()
        market_state = mse_engine.market_state()
        prediction = engine.predict(market_state, symbol="EUR_USD", timeframe="M5")
        vec, names = market_state.to_vector()
        snapshot = ModelMonitor.from_regression_prediction(
            prediction, model_name=monitor.model_name, primary_target="next_close",
            decision_index=decision_idx, feature_vector=vec, feature_names=names,
        )
        monitor.record_prediction(snapshot)


def _live_predictions_classification(engine, df, window_size, n_live, monitor):
    start = len(df) - window_size - 5 - n_live
    for i in range(n_live):
        decision_idx = start + i + window_size - 1
        window = df.iloc[start + i: start + i + window_size]
        mse_engine = MarketStructureEngine(EngineConfig(swing_window=5))
        mse_engine.load(window)
        mse_engine.analyze()
        market_state = mse_engine.market_state()
        prediction = engine.predict(market_state, symbol="EUR_USD", timeframe="M5")
        vec, names = market_state.to_vector()
        snapshot = ModelMonitor.from_classification_prediction(
            prediction, model_name=monitor.model_name, decision_index=decision_idx,
            feature_vector=vec, feature_names=names,
        )
        monitor.record_prediction(snapshot)


def test_full_regression_lifecycle_with_real_engine(tmp_path, mon_regression_engine) -> None:
    engine, dataset, split, ml_cfg, records = mon_regression_engine
    df = make_ohlcv(700, seed=17)

    config = MonitorConfig(min_new_samples=5, rolling_window=10, market_regime_lookback=20, max_model_age_days=9999)
    monitor = ModelMonitor(config, model_name="lr_test_close", task_type="regression", output_root=tmp_path)

    record = records["next_close"]
    lifecycle = ModelLifecycleMetadata(
        model_name="lr_test_close", version=record.experiment_id, task_type="regression", status="candidate",
        training_date=record.timestamp, training_dataset_size=len(dataset),
        feature_version="1.0.0", training_version="1.0.0", strategy_version="1.0.0", dataset_version="1.0.0",
    )
    monitor.register_model(lifecycle, as_production=True)
    assert monitor.lifecycle_registry.production_version("lr_test_close").version == record.experiment_id

    X_train = dataset.X[split.train_idx]
    monitor.fit_baseline(X_train, dataset.feature_names, target_values=dataset.y_reg["next_close"][split.train_idx])

    _live_predictions_regression(engine, df, ml_cfg.window_size, 35, monitor)
    assert len(monitor.prediction_log) == 35

    resolver = functools.partial(REGRESSION_REGISTRY["next_close"], pip_size=0.0001)
    resolved = monitor.resolve_outcomes(df, resolver)
    assert len(resolved) > 0
    assert monitor.prediction_log.coverage() == 1.0

    health = monitor.evaluate_health()
    assert 0.0 <= health.health.overall <= 100.0
    assert health.status in ("good", "warning", "critical")
    assert health.task_type == "regression"

    rec = monitor.recommend_retraining(current_horizon=5)
    assert rec.priority in ("none", "low", "medium", "high", "critical")

    report = monitor.to_agentic_report(model_display_name="Linear Regression")
    assert set(report) == {
        "model", "health_score", "status", "feature_drift", "calibration",
        "retraining_recommended", "estimated_improvement", "priority",
    }
    assert report["model"] == "Linear Regression"
    assert isinstance(report["health_score"], int)
    assert isinstance(report["retraining_recommended"], bool)

    # Manual mode: check_retraining never trains, only notifies.
    outcome = monitor.check_retraining(current_horizon=5)
    assert outcome.triggered is False


def test_full_classification_lifecycle_with_real_engine(tmp_path, mon_classification_engine) -> None:
    engine, dataset, split, ml_cfg, record = mon_classification_engine
    df = make_ohlcv(700, seed=17)

    config = MonitorConfig(min_new_samples=5, rolling_window=10, market_regime_lookback=20, max_model_age_days=9999)
    monitor = ModelMonitor(
        config, model_name="lgr_test", task_type="classification", output_root=tmp_path,
        classes=["SELL", "NO_TRADE", "BUY"],
    )

    lifecycle = ModelLifecycleMetadata(
        model_name="lgr_test", version=record.experiment_id, task_type="classification", status="candidate",
        training_date=record.timestamp, training_dataset_size=len(dataset),
        feature_version="1.0.0", training_version="1.0.0", strategy_version="1.0.0", dataset_version="1.0.0",
    )
    monitor.register_model(lifecycle, as_production=True)

    X_train = dataset.X[split.train_idx]
    monitor.fit_baseline(X_train, dataset.feature_names)

    _live_predictions_classification(engine, df, ml_cfg.window_size, 35, monitor)

    from logistic_regression.label_manager import ConfigurableClassificationLabelGenerator
    label_gen = ConfigurableClassificationLabelGenerator(min_pip_movement=5.0, risk_reward_threshold=1.5)
    resolved = monitor.resolve_outcomes(df, label_gen.label)
    assert len(resolved) > 0

    health = monitor.evaluate_health()
    assert health.task_type == "classification"
    assert 0.0 <= health.health.overall <= 100.0

    rec = monitor.recommend_retraining(current_horizon=5)
    report = monitor.to_agentic_report(model_display_name="Logistic Regression")
    assert report["model"] == "Logistic Regression"


def test_evaluate_health_before_fit_baseline_raises(tmp_path) -> None:
    monitor = ModelMonitor(MonitorConfig(), model_name="m", task_type="regression", output_root=tmp_path)
    with pytest.raises(InsufficientDataError):
        monitor.evaluate_health()


def test_to_agentic_report_before_evaluate_health_raises(tmp_path) -> None:
    monitor = ModelMonitor(MonitorConfig(), model_name="m", task_type="regression", output_root=tmp_path)
    with pytest.raises(InsufficientDataError):
        monitor.to_agentic_report()


def test_unknown_task_type_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        ModelMonitor(MonitorConfig(), model_name="m", task_type="bogus", output_root=tmp_path)


def test_record_prediction_validates_by_default(tmp_path) -> None:
    from model_monitor.prediction_monitor import PredictionSnapshot

    monitor = ModelMonitor(MonitorConfig(), model_name="m", task_type="regression", output_root=tmp_path)
    bad_snapshot = PredictionSnapshot(
        task_type="regression", model_name="m", model_version="1", feature_version="1",
        training_version="1", symbol="EUR_USD", timeframe="M5", prediction_horizon=5,
        timestamp="t", decision_index=0, feature_vector=[1.0, 2.0], feature_names=["a", "b"],
        confidence=500.0, predicted_value=1.0, raw_predictions={},  # confidence way out of range
    )
    with pytest.raises(ModelMonitorError):
        monitor.record_prediction(bad_snapshot)


def test_evaluate_health_uses_specific_model_version(tmp_path, mon_regression_engine) -> None:
    engine, dataset, split, ml_cfg, records = mon_regression_engine
    df = make_ohlcv(700, seed=17)
    config = MonitorConfig(min_new_samples=5, rolling_window=10, market_regime_lookback=20, max_model_age_days=9999)
    monitor = ModelMonitor(config, model_name="lr_test_close_v2", task_type="regression", output_root=tmp_path)

    record = records["next_close"]
    lifecycle = ModelLifecycleMetadata(
        model_name="lr_test_close_v2", version="explicit_v1", task_type="regression", status="candidate",
        training_date=record.timestamp, training_dataset_size=len(dataset),
        feature_version="1.0.0", training_version="1.0.0", strategy_version="1.0.0", dataset_version="1.0.0",
    )
    monitor.register_model(lifecycle, as_production=True)
    X_train = dataset.X[split.train_idx]
    monitor.fit_baseline(X_train, dataset.feature_names, target_values=dataset.y_reg["next_close"][split.train_idx])
    _live_predictions_regression(engine, df, ml_cfg.window_size, 20, monitor)
    resolver = functools.partial(REGRESSION_REGISTRY["next_close"], pip_size=0.0001)
    monitor.resolve_outcomes(df, resolver)

    health = monitor.evaluate_health(model_version="explicit_v1")
    assert health.model_version == "explicit_v1"
