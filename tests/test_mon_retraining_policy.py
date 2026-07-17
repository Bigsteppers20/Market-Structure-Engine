"""Tests for model_monitor.retraining_policy.evaluate_retraining_policy --
RETRAINING RECOMMENDATION section."""
from __future__ import annotations

import numpy as np
import pytest

from model_monitor.calibration_monitor import CalibrationMonitor
from model_monitor.config import MonitorConfig
from model_monitor.drift_detector import DriftDetector
from model_monitor.health_engine import HealthEngine
from model_monitor.model_registry import ModelLifecycleMetadata
from model_monitor.prediction_monitor import PredictionLog, PredictionSnapshot, ResolvedPrediction
from model_monitor.retraining_policy import evaluate_retraining_policy

NAMES = [f"f{i}" for i in range(5)]
NOW = "2026-07-01T00:00:00+00:00"


def _lifecycle(training_date: str = "2026-06-25T00:00:00+00:00", dataset_size: int = 1000) -> ModelLifecycleMetadata:
    return ModelLifecycleMetadata(
        model_name="m", version="1", task_type="regression", status="production",
        training_date=training_date, training_dataset_size=dataset_size,
        feature_version="1", training_version="1", strategy_version="1", dataset_version="1",
    )


def _regression_log(n: int = 150, seed: int = 0, noise: float = 0.0001) -> PredictionLog:
    rng = np.random.default_rng(seed)
    log = PredictionLog()
    for i in range(n):
        pred = rng.normal(0, 0.001)
        actual = pred + rng.normal(0, noise)
        snap = PredictionSnapshot(
            task_type="regression", model_name="m", model_version="1", feature_version="1",
            training_version="1", symbol="EUR_USD", timeframe="M5", prediction_horizon=5,
            timestamp="t", decision_index=i, feature_vector=list(rng.normal(0, 1, 5)), feature_names=NAMES,
            confidence=rng.uniform(40, 90), predicted_value=pred, raw_predictions={},
        )
        log._resolved.append(ResolvedPrediction(snapshot=snap, resolved_at="t", actual_value=actual))
    return log


def _health_and_drift(*, config: MonitorConfig, log_noise: float = 0.0001, drift_shift: float = 0.0, lifecycle=None):
    log = _regression_log(noise=log_noise)
    cal_report = CalibrationMonitor().evaluate(log.resolved)
    rng = np.random.default_rng(0)
    detector = DriftDetector(config).fit_baseline(rng.normal(0, 1, (300, 5)), NAMES)
    drift_report = detector.detect(rng.normal(drift_shift, 1, (50, 5)), NAMES)
    lifecycle = lifecycle or _lifecycle()
    health_report = HealthEngine(config).evaluate(
        model_name="m", model_version="1", task_type="regression", prediction_log=log,
        drift_report=drift_report, calibration_report=cal_report, lifecycle=lifecycle, now_iso=NOW,
    )
    return health_report, drift_report, lifecycle


def test_healthy_model_not_recommended() -> None:
    config = MonitorConfig(health_threshold=1.0, feature_drift_threshold=0.99, max_model_age_days=9999)
    health_report, drift_report, lifecycle = _health_and_drift(config=config)
    rec = evaluate_retraining_policy(
        health_report=health_report, drift_report=drift_report, lifecycle=lifecycle,
        config=config, new_sample_count=1000, now_iso=NOW, current_horizon=5,
    )
    assert rec.recommended is False
    assert rec.priority == "none"
    assert rec.reasons == []


def test_low_health_score_triggers_recommendation() -> None:
    config = MonitorConfig(health_threshold=99.9, feature_drift_threshold=0.99, max_model_age_days=9999)
    health_report, drift_report, lifecycle = _health_and_drift(config=config)
    rec = evaluate_retraining_policy(
        health_report=health_report, drift_report=drift_report, lifecycle=lifecycle,
        config=config, new_sample_count=1000, now_iso=NOW, current_horizon=5,
    )
    assert rec.recommended is True
    assert any("Health score" in r for r in rec.reasons)


def test_feature_drift_triggers_recommendation() -> None:
    config = MonitorConfig(health_threshold=1.0, feature_drift_threshold=0.05, max_model_age_days=9999)
    health_report, drift_report, lifecycle = _health_and_drift(config=config, drift_shift=3.0)
    rec = evaluate_retraining_policy(
        health_report=health_report, drift_report=drift_report, lifecycle=lifecycle,
        config=config, new_sample_count=1000, now_iso=NOW, current_horizon=5,
    )
    assert rec.recommended is True
    assert any("Feature drift" in r for r in rec.reasons)


def test_model_age_triggers_recommendation() -> None:
    config = MonitorConfig(health_threshold=1.0, feature_drift_threshold=0.99, max_model_age_days=10.0)
    health_report, drift_report, lifecycle = _health_and_drift(
        config=config, lifecycle=_lifecycle(training_date="2026-01-01T00:00:00+00:00"),
    )
    rec = evaluate_retraining_policy(
        health_report=health_report, drift_report=drift_report, lifecycle=lifecycle,
        config=config, new_sample_count=1000, now_iso=NOW, current_horizon=5,
    )
    assert rec.recommended is True
    assert any("Model age" in r for r in rec.reasons)


def test_insufficient_data_defers_recommendation() -> None:
    config = MonitorConfig(health_threshold=99.9, feature_drift_threshold=0.99, max_model_age_days=9999, min_new_samples=5000)
    health_report, drift_report, lifecycle = _health_and_drift(config=config)
    rec = evaluate_retraining_policy(
        health_report=health_report, drift_report=drift_report, lifecycle=lifecycle,
        config=config, new_sample_count=10, now_iso=NOW, current_horizon=5,
    )
    assert rec.recommended is False  # condition true, but not enough data
    assert any("Insufficient new data" in r for r in rec.reasons)


def test_priority_escalates_with_number_of_triggers() -> None:
    lenient = MonitorConfig(health_threshold=1.0, feature_drift_threshold=0.99, max_model_age_days=9999)
    strict = MonitorConfig(health_threshold=99.9, feature_drift_threshold=0.01, max_model_age_days=10.0)
    health_lenient, drift_lenient, lc_lenient = _health_and_drift(config=lenient)
    rec_lenient = evaluate_retraining_policy(
        health_report=health_lenient, drift_report=drift_lenient, lifecycle=lc_lenient,
        config=lenient, new_sample_count=1000, now_iso=NOW, current_horizon=5,
    )
    health_strict, drift_strict, lc_strict = _health_and_drift(
        config=strict, drift_shift=3.0, lifecycle=_lifecycle(training_date="2026-01-01T00:00:00+00:00"),
    )
    rec_strict = evaluate_retraining_policy(
        health_report=health_strict, drift_report=drift_strict, lifecycle=lc_strict,
        config=strict, new_sample_count=1000, now_iso=NOW, current_horizon=5,
    )
    priority_rank = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    assert priority_rank[rec_strict.priority] > priority_rank[rec_lenient.priority]


def test_suggested_horizon_matches_current_horizon() -> None:
    config = MonitorConfig(health_threshold=99.9, max_model_age_days=9999)
    health_report, drift_report, lifecycle = _health_and_drift(config=config)
    rec = evaluate_retraining_policy(
        health_report=health_report, drift_report=drift_report, lifecycle=lifecycle,
        config=config, new_sample_count=1000, now_iso=NOW, current_horizon=10,
    )
    assert rec.suggested_horizon == 10


def test_suggested_dataset_size_at_least_min_new_samples() -> None:
    config = MonitorConfig(health_threshold=1.0, max_model_age_days=9999, min_new_samples=5000)
    health_report, drift_report, lifecycle = _health_and_drift(config=config, lifecycle=_lifecycle(dataset_size=100))
    rec = evaluate_retraining_policy(
        health_report=health_report, drift_report=drift_report, lifecycle=lifecycle,
        config=config, new_sample_count=1000, now_iso=NOW, current_horizon=5,
    )
    assert rec.suggested_dataset_size >= 5000


def test_recommendation_to_dict_serializable() -> None:
    config = MonitorConfig(health_threshold=1.0, max_model_age_days=9999)
    health_report, drift_report, lifecycle = _health_and_drift(config=config)
    rec = evaluate_retraining_policy(
        health_report=health_report, drift_report=drift_report, lifecycle=lifecycle,
        config=config, new_sample_count=1000, now_iso=NOW, current_horizon=5,
    )
    d = rec.to_dict()
    assert set(d) == {"recommended", "priority", "reasons", "estimated_benefit", "suggested_dataset_size", "suggested_horizon"}
