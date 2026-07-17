"""Tests for model_monitor.calibration_monitor -- CALIBRATION MONITOR
section: overconfidence/underconfidence/calibration drift, working
identically for regression and classification resolved predictions."""
from __future__ import annotations

import numpy as np
import pytest

from model_monitor.calibration_monitor import CalibrationMonitor, detect_calibration_drift
from model_monitor.exceptions import InsufficientDataError
from model_monitor.prediction_monitor import PredictionSnapshot, ResolvedPrediction


def _classification_resolved(n: int, *, confidence_fn, correct_fn, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        conf = confidence_fn(rng)
        correct = correct_fn(rng, conf)
        snap = PredictionSnapshot(
            task_type="classification", model_name="m", model_version="1", feature_version="1",
            training_version="1", symbol="EUR_USD", timeframe="M5", prediction_horizon=5,
            timestamp="t", decision_index=i, feature_vector=[0.0], feature_names=["a"], confidence=conf,
            predicted_class="BUY", class_probabilities={"BUY": conf / 100, "SELL": 0.0, "NO_TRADE": 1 - conf / 100},
        )
        out.append(ResolvedPrediction(snapshot=snap, resolved_at="t", actual_class="BUY" if correct else "SELL"))
    return out


def test_evaluate_needs_at_least_one_resolved_outcome() -> None:
    with pytest.raises(InsufficientDataError):
        CalibrationMonitor().evaluate([])


def test_well_calibrated_model() -> None:
    resolved = _classification_resolved(
        300, confidence_fn=lambda rng: rng.uniform(50, 95),
        correct_fn=lambda rng, conf: rng.random() < conf / 100.0,
    )
    report = CalibrationMonitor().evaluate(resolved)
    assert report.status == "well_calibrated"
    assert report.calibration_error < 0.15


def test_overconfident_model() -> None:
    resolved = _classification_resolved(
        300, confidence_fn=lambda rng: rng.uniform(80, 99),
        correct_fn=lambda rng, conf: rng.random() < 0.3,  # confident but usually wrong
    )
    report = CalibrationMonitor().evaluate(resolved)
    assert report.status == "overconfident"
    assert report.overall_mean_confidence > report.overall_observed_accuracy


def test_underconfident_model() -> None:
    resolved = _classification_resolved(
        300, confidence_fn=lambda rng: rng.uniform(20, 40),
        correct_fn=lambda rng, conf: rng.random() < 0.9,  # unconfident but usually right
    )
    report = CalibrationMonitor().evaluate(resolved)
    assert report.status == "underconfident"
    assert report.overall_observed_accuracy > report.overall_mean_confidence


def test_regression_calibration_uses_directional_correctness() -> None:
    rng = np.random.default_rng(1)
    resolved = []
    for i in range(200):
        conf = rng.uniform(40, 90)
        pred = rng.normal(0, 0.001)
        # Correct direction more often when confidence is high.
        actual = pred if rng.random() < conf / 100.0 else -pred
        snap = PredictionSnapshot(
            task_type="regression", model_name="m", model_version="1", feature_version="1",
            training_version="1", symbol="EUR_USD", timeframe="M5", prediction_horizon=5,
            timestamp="t", decision_index=i, feature_vector=[0.0], feature_names=["a"], confidence=conf,
            predicted_value=pred, raw_predictions={}, primary_target="next_return",
        )
        resolved.append(ResolvedPrediction(snapshot=snap, resolved_at="t", actual_value=actual))
    report = CalibrationMonitor().evaluate(resolved)
    assert 0.0 <= report.calibration_error <= 1.0
    assert report.n_samples == 200


def test_buckets_partition_by_confidence_range() -> None:
    resolved = _classification_resolved(
        200, confidence_fn=lambda rng: rng.uniform(0, 100), correct_fn=lambda rng, conf: rng.random() < 0.5,
    )
    report = CalibrationMonitor(n_buckets=5).evaluate(resolved)
    for bucket in report.buckets:
        assert bucket.n_samples > 0
        lo, hi = bucket.bucket_range
        assert lo < hi


def test_detect_calibration_drift() -> None:
    baseline = _classification_resolved(
        200, confidence_fn=lambda rng: rng.uniform(50, 90), correct_fn=lambda rng, conf: rng.random() < conf / 100.0, seed=1,
    )
    drifted = _classification_resolved(
        200, confidence_fn=lambda rng: rng.uniform(80, 99), correct_fn=lambda rng, conf: rng.random() < 0.2, seed=2,
    )
    mon = CalibrationMonitor()
    baseline_report = mon.evaluate(baseline)
    current_report = mon.evaluate(drifted)
    drift = detect_calibration_drift(baseline_report, current_report)
    assert drift > 0.1
