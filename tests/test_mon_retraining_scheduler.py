"""Tests for model_monitor.retraining_scheduler -- RETRAINING MODES section:
manual (notify-only), scheduled (time-based), adaptive (strict AND-gate)."""
from __future__ import annotations

import numpy as np
import pytest

from model_monitor.calibration_monitor import CalibrationMonitor
from model_monitor.config import MonitorConfig, RetrainingScheduleConfig
from model_monitor.drift_detector import DriftDetector
from model_monitor.health_engine import HealthEngine
from model_monitor.model_registry import ModelLifecycleMetadata
from model_monitor.prediction_monitor import PredictionLog, PredictionSnapshot, ResolvedPrediction
from model_monitor.retraining_scheduler import is_due, should_trigger_retraining

NAMES = [f"f{i}" for i in range(5)]
NOW = "2026-07-01T00:00:00+00:00"


def _lifecycle(training_date: str) -> ModelLifecycleMetadata:
    return ModelLifecycleMetadata(
        model_name="m", version="1", task_type="regression", status="production",
        training_date=training_date, training_dataset_size=1000,
        feature_version="1", training_version="1", strategy_version="1", dataset_version="1",
    )


def _health_and_drift(*, config: MonitorConfig, drift_shift: float = 0.0):
    rng = np.random.default_rng(0)
    log = PredictionLog()
    for i in range(150):
        pred = rng.normal(0, 0.001)
        actual = pred + rng.normal(0, 0.0001)
        snap = PredictionSnapshot(
            task_type="regression", model_name="m", model_version="1", feature_version="1",
            training_version="1", symbol="EUR_USD", timeframe="M5", prediction_horizon=5,
            timestamp="t", decision_index=i, feature_vector=list(rng.normal(0, 1, 5)), feature_names=NAMES,
            confidence=rng.uniform(40, 90), predicted_value=pred, raw_predictions={},
        )
        log._resolved.append(ResolvedPrediction(snapshot=snap, resolved_at="t", actual_value=actual))
    cal_report = CalibrationMonitor().evaluate(log.resolved)
    detector = DriftDetector(config).fit_baseline(rng.normal(0, 1, (300, 5)), NAMES)
    drift_report = detector.detect(rng.normal(drift_shift, 1, (50, 5)), NAMES)
    lifecycle = _lifecycle("2026-06-25T00:00:00+00:00")
    health_report = HealthEngine(config).evaluate(
        model_name="m", model_version="1", task_type="regression", prediction_log=log,
        drift_report=drift_report, calibration_report=cal_report, lifecycle=lifecycle, now_iso=NOW,
    )
    return health_report, drift_report


# --------------------------------------------------------------------------- #
# is_due
# --------------------------------------------------------------------------- #
def test_is_due_true_after_interval_elapsed() -> None:
    schedule = RetrainingScheduleConfig(frequency="weekly")
    assert is_due(schedule, "2026-06-01T00:00:00+00:00", "2026-06-10T00:00:00+00:00")


def test_is_due_false_before_interval_elapsed() -> None:
    schedule = RetrainingScheduleConfig(frequency="weekly")
    assert not is_due(schedule, "2026-06-01T00:00:00+00:00", "2026-06-03T00:00:00+00:00")


# --------------------------------------------------------------------------- #
# should_trigger_retraining
# --------------------------------------------------------------------------- #
def test_manual_mode_never_triggers() -> None:
    config = MonitorConfig(health_threshold=99.9, feature_drift_threshold=0.01, auto_retraining_enabled=True)
    health_report, drift_report = _health_and_drift(config=config, drift_shift=3.0)
    triggered, reason = should_trigger_retraining(
        mode="manual", schedule=config.retraining_schedule, health_report=health_report,
        drift_report=drift_report, config=config, last_trained_iso="2026-01-01T00:00:00+00:00",
        now_iso=NOW, new_sample_count=10000,
    )
    assert triggered is False
    assert "Manual" in reason


def test_scheduled_mode_triggers_when_due() -> None:
    config = MonitorConfig(retraining_schedule=RetrainingScheduleConfig(frequency="daily"))
    health_report, drift_report = _health_and_drift(config=config)
    triggered, reason = should_trigger_retraining(
        mode="scheduled", schedule=config.retraining_schedule, health_report=health_report,
        drift_report=drift_report, config=config, last_trained_iso="2026-06-01T00:00:00+00:00",
        now_iso=NOW, new_sample_count=1,
    )
    assert triggered is True
    assert "due" in reason.lower()


def test_scheduled_mode_does_not_trigger_when_not_due() -> None:
    config = MonitorConfig(retraining_schedule=RetrainingScheduleConfig(frequency="monthly"))
    health_report, drift_report = _health_and_drift(config=config)
    triggered, reason = should_trigger_retraining(
        mode="scheduled", schedule=config.retraining_schedule, health_report=health_report,
        drift_report=drift_report, config=config, last_trained_iso="2026-06-30T00:00:00+00:00",
        now_iso=NOW, new_sample_count=1,
    )
    assert triggered is False


def test_adaptive_mode_requires_policy_permission() -> None:
    config = MonitorConfig(auto_retraining_enabled=False, health_threshold=99.9, feature_drift_threshold=0.01, min_new_samples=1)
    health_report, drift_report = _health_and_drift(config=config, drift_shift=3.0)
    triggered, reason = should_trigger_retraining(
        mode="adaptive", schedule=config.retraining_schedule, health_report=health_report,
        drift_report=drift_report, config=config, last_trained_iso="2026-01-01T00:00:00+00:00",
        now_iso=NOW, new_sample_count=1000,
    )
    assert triggered is False
    assert "auto_retraining_enabled=False" in reason


def test_adaptive_mode_triggers_only_when_all_conditions_met() -> None:
    config = MonitorConfig(
        auto_retraining_enabled=True, health_threshold=99.9, feature_drift_threshold=0.01, min_new_samples=100,
    )
    health_report, drift_report = _health_and_drift(config=config, drift_shift=3.0)
    triggered, reason = should_trigger_retraining(
        mode="adaptive", schedule=config.retraining_schedule, health_report=health_report,
        drift_report=drift_report, config=config, last_trained_iso="2026-01-01T00:00:00+00:00",
        now_iso=NOW, new_sample_count=1000,
    )
    assert triggered is True
    assert "Adaptive conditions met" in reason


def test_adaptive_mode_does_not_trigger_on_partial_conditions() -> None:
    """Health low but drift NOT above threshold -- must not trigger (the
    spec's adaptive gate is a strict AND, not an OR)."""
    config = MonitorConfig(
        auto_retraining_enabled=True, health_threshold=99.9, feature_drift_threshold=0.99, min_new_samples=100,
    )
    health_report, drift_report = _health_and_drift(config=config, drift_shift=0.0)
    triggered, reason = should_trigger_retraining(
        mode="adaptive", schedule=config.retraining_schedule, health_report=health_report,
        drift_report=drift_report, config=config, last_trained_iso="2026-01-01T00:00:00+00:00",
        now_iso=NOW, new_sample_count=1000,
    )
    assert triggered is False
    assert "feature drift not above threshold" in reason


def test_adaptive_mode_does_not_trigger_on_insufficient_data() -> None:
    config = MonitorConfig(
        auto_retraining_enabled=True, health_threshold=99.9, feature_drift_threshold=0.01, min_new_samples=10000,
    )
    health_report, drift_report = _health_and_drift(config=config, drift_shift=3.0)
    triggered, reason = should_trigger_retraining(
        mode="adaptive", schedule=config.retraining_schedule, health_report=health_report,
        drift_report=drift_report, config=config, last_trained_iso="2026-01-01T00:00:00+00:00",
        now_iso=NOW, new_sample_count=5,
    )
    assert triggered is False
    assert "insufficient new data" in reason.lower()


def test_unknown_mode_raises() -> None:
    config = MonitorConfig()
    health_report, drift_report = _health_and_drift(config=config)
    with pytest.raises(ValueError):
        should_trigger_retraining(
            mode="bogus", schedule=config.retraining_schedule, health_report=health_report,
            drift_report=drift_report, config=config, last_trained_iso=NOW, now_iso=NOW, new_sample_count=1,
        )
