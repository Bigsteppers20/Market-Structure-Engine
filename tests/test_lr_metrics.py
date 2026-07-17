"""Tests for linear_regression.metrics."""
from __future__ import annotations

import numpy as np
import pytest

from linear_regression.metrics import (
    compute_all_regression_metrics,
    prediction_error_distribution,
    residual_statistics,
)
from training.metrics import METRIC_REGISTRY


def test_explained_variance_registered_in_shared_training_registry() -> None:
    assert "explained_variance" in METRIC_REGISTRY
    assert METRIC_REGISTRY["explained_variance"].task_type == "regression"


def test_compute_all_regression_metrics_includes_base_and_new_fields() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    out = compute_all_regression_metrics(y_true, y_pred)
    for key in ("mae", "mse", "rmse", "r2", "mape", "explained_variance",
                "residual_statistics", "prediction_error_distribution"):
        assert key in out


def test_residual_statistics_perfect_prediction_is_zero() -> None:
    y = np.array([1.0, 2.0, 3.0])
    stats = residual_statistics(y, y)
    assert stats["mean"] == pytest.approx(0.0)
    assert stats["std"] == pytest.approx(0.0)


def test_residual_statistics_empty() -> None:
    stats = residual_statistics(np.array([]), np.array([]))
    assert stats["mean"] == 0.0


def test_prediction_error_distribution_shape() -> None:
    y_true = np.linspace(0, 1, 50)
    y_pred = y_true + np.random.default_rng(0).normal(scale=0.05, size=50)
    dist = prediction_error_distribution(y_true, y_pred, bins=5)
    assert len(dist["counts"]) == 5
    assert len(dist["bin_edges"]) == 6
    assert sum(dist["counts"]) == 50
