"""Tests for training.metrics."""
from __future__ import annotations

import numpy as np
import pytest

from training.metrics import (
    METRIC_REGISTRY,
    InferenceStatistics,
    Metric,
    TrainingStatistics,
    compute_classification_metrics,
    compute_regression_metrics,
    metrics_for_task,
    register_metric,
)


def test_regression_metrics_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = compute_regression_metrics(y, y)
    assert out["mae"] == pytest.approx(0.0)
    assert out["mse"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(0.0)
    assert out["r2"] == pytest.approx(1.0)


def test_regression_metrics_known_values() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    out = compute_regression_metrics(y_true, y_pred)
    assert out["mae"] == pytest.approx(2.0 / 3.0)


def test_classification_metrics_perfect_prediction() -> None:
    y = np.array([0, 1, 2, 1, 0])
    out = compute_classification_metrics(y, y)
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["f1"] == pytest.approx(1.0)
    cm = np.array(out["confusion_matrix"])
    assert cm.sum() == len(y)
    assert np.trace(cm) == len(y)  # all correct -> diagonal


def test_classification_metrics_imperfect() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    out = compute_classification_metrics(y_true, y_pred)
    assert 0.0 < out["accuracy"] < 1.0


def test_metrics_for_task_partitions_registry() -> None:
    reg = set(metrics_for_task("regression"))
    cls = set(metrics_for_task("classification"))
    assert reg & cls == set()
    assert "mae" in reg
    assert "accuracy" in cls


def test_register_metric_extends_without_modifying_existing_code() -> None:
    class Median(Metric):
        name, task_type = "custom_median_ae", "regression"

        def compute(self, y_true, y_pred, **kwargs):
            return float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    register_metric(Median())
    try:
        assert "custom_median_ae" in metrics_for_task("regression")
        out = compute_regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 4.0]))
        assert "custom_median_ae" in out
    finally:
        del METRIC_REGISTRY["custom_median_ae"]  # keep test isolated


def test_training_statistics_to_dict() -> None:
    stats = TrainingStatistics(
        duration_seconds=1.5, n_train_samples=100, n_val_samples=20, n_test_samples=20,
        n_features=185, random_seed=42, started_at="t0", finished_at="t1",
    )
    d = stats.to_dict()
    assert d["n_features"] == 185
    assert d["duration_seconds"] == 1.5


def test_inference_statistics_from_latencies() -> None:
    stats = InferenceStatistics.from_latencies([10.0, 20.0, 30.0, 40.0, 50.0])
    assert stats.n_predictions == 5
    assert stats.mean_latency_ms == pytest.approx(30.0)
    assert stats.p50_latency_ms == pytest.approx(30.0)


def test_inference_statistics_empty() -> None:
    stats = InferenceStatistics.from_latencies([])
    assert stats.n_predictions == 0
    assert stats.mean_latency_ms == 0.0
