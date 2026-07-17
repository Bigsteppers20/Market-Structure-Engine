"""Tests for model_monitor.health_engine.HealthEngine -- orchestrates
performance/calibration/drift/lifecycle into one ModelHealthReport."""
from __future__ import annotations

import numpy as np
import pytest

from model_monitor.calibration_monitor import CalibrationMonitor
from model_monitor.config import MonitorConfig
from model_monitor.drift_detector import DriftDetector
from model_monitor.exceptions import InsufficientDataError
from model_monitor.health_engine import HealthEngine
from model_monitor.model_registry import ModelLifecycleMetadata
from model_monitor.prediction_monitor import PredictionLog, PredictionSnapshot, ResolvedPrediction

NAMES = [f"f{i}" for i in range(5)]


def _lifecycle(training_date: str = "2026-06-01T00:00:00+00:00") -> ModelLifecycleMetadata:
    return ModelLifecycleMetadata(
        model_name="m", version="1", task_type="regression", status="production",
        training_date=training_date, training_dataset_size=1000,
        feature_version="1", training_version="1", strategy_version="1", dataset_version="1",
    )


def _regression_log(n: int = 120, seed: int = 0, noise: float = 0.0003) -> PredictionLog:
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


def _drift_report(seed: int = 0):
    rng = np.random.default_rng(seed)
    detector = DriftDetector().fit_baseline(rng.normal(0, 1, (300, 5)), NAMES)
    return detector.detect(rng.normal(0, 1, (50, 5)), NAMES)


def test_evaluate_needs_resolved_predictions() -> None:
    engine = HealthEngine(MonitorConfig())
    with pytest.raises(InsufficientDataError):
        engine.evaluate(
            model_name="m", model_version="1", task_type="regression", prediction_log=PredictionLog(),
            drift_report=_drift_report(), calibration_report=CalibrationMonitor().evaluate(_regression_log().resolved),
            lifecycle=_lifecycle(), now_iso="2026-07-01T00:00:00+00:00",
        )


def test_healthy_model_reports_good_status() -> None:
    log = _regression_log(150, noise=0.0001)  # tight predictions -> good R^2
    cal_report = CalibrationMonitor().evaluate(log.resolved)
    drift_report = _drift_report()
    lifecycle = _lifecycle(training_date="2026-06-25T00:00:00+00:00")  # recent
    engine = HealthEngine(MonitorConfig(max_model_age_days=365))
    report = engine.evaluate(
        model_name="m", model_version="1", task_type="regression", prediction_log=log,
        drift_report=drift_report, calibration_report=cal_report, lifecycle=lifecycle,
        now_iso="2026-07-01T00:00:00+00:00",
    )
    assert report.status in ("good", "warning")
    assert 0.0 <= report.health.overall <= 100.0


def test_degraded_model_reports_lower_health_than_healthy_model() -> None:
    healthy_log = _regression_log(150, seed=1, noise=0.00005)
    degraded_log = _regression_log(150, seed=1, noise=0.01)  # far noisier predictions
    cal = CalibrationMonitor()
    drift_report = _drift_report()
    lifecycle = _lifecycle(training_date="2026-06-25T00:00:00+00:00")
    engine = HealthEngine(MonitorConfig(max_model_age_days=365))

    healthy = engine.evaluate(
        model_name="m", model_version="1", task_type="regression", prediction_log=healthy_log,
        drift_report=drift_report, calibration_report=cal.evaluate(healthy_log.resolved),
        lifecycle=lifecycle, now_iso="2026-07-01T00:00:00+00:00",
    )
    degraded = engine.evaluate(
        model_name="m", model_version="1", task_type="regression", prediction_log=degraded_log,
        drift_report=drift_report, calibration_report=cal.evaluate(degraded_log.resolved),
        lifecycle=lifecycle, now_iso="2026-07-01T00:00:00+00:00",
    )
    assert degraded.health.overall < healthy.health.overall
    assert degraded.health.prediction_accuracy <= healthy.health.prediction_accuracy


def test_old_model_scores_lower_training_age() -> None:
    log = _regression_log(120)
    cal_report = CalibrationMonitor().evaluate(log.resolved)
    drift_report = _drift_report()
    engine = HealthEngine(MonitorConfig(max_model_age_days=30))

    fresh = engine.evaluate(
        model_name="m", model_version="1", task_type="regression", prediction_log=log,
        drift_report=drift_report, calibration_report=cal_report,
        lifecycle=_lifecycle(training_date="2026-06-30T00:00:00+00:00"),
        now_iso="2026-07-01T00:00:00+00:00",
    )
    stale = engine.evaluate(
        model_name="m", model_version="1", task_type="regression", prediction_log=log,
        drift_report=drift_report, calibration_report=cal_report,
        lifecycle=_lifecycle(training_date="2026-01-01T00:00:00+00:00"),
        now_iso="2026-07-01T00:00:00+00:00",
    )
    assert stale.health.training_age < fresh.health.training_age
    assert stale.health.training_age == 0.0  # far past max_model_age_days


def test_report_to_dict_serializable() -> None:
    log = _regression_log(120)
    cal_report = CalibrationMonitor().evaluate(log.resolved)
    drift_report = _drift_report()
    engine = HealthEngine(MonitorConfig())
    report = engine.evaluate(
        model_name="m", model_version="1", task_type="regression", prediction_log=log,
        drift_report=drift_report, calibration_report=cal_report, lifecycle=_lifecycle(),
        now_iso="2026-07-01T00:00:00+00:00",
    )
    d = report.to_dict()
    assert d["model_name"] == "m"
    assert "health" in d and "drift" in d and "performance" in d
