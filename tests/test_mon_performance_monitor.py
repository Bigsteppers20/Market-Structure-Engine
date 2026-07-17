"""Tests for model_monitor.performance_monitor -- PERFORMANCE MONITORING
section, reusing linear_regression/logistic_regression metrics unmodified."""
from __future__ import annotations

import numpy as np
import pytest

from model_monitor.exceptions import InsufficientDataError
from model_monitor.performance_monitor import PerformanceMonitor
from model_monitor.prediction_monitor import PredictionSnapshot, ResolvedPrediction


def _regression_resolved(n: int, seed: int = 0, noise: float = 0.0003):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        pred = rng.normal(0, 0.001)
        actual = pred + rng.normal(0, noise)
        snap = PredictionSnapshot(
            task_type="regression", model_name="m", model_version="1", feature_version="1",
            training_version="1", symbol="EUR_USD", timeframe="M5", prediction_horizon=5,
            timestamp="t", decision_index=i, feature_vector=[0.0], feature_names=["a"], confidence=60.0,
            predicted_value=pred, raw_predictions={"next_return": pred}, primary_target="next_return",
        )
        out.append(ResolvedPrediction(snapshot=snap, resolved_at="t", actual_value=actual))
    return out


def _classification_resolved(n: int, classes, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        proba = rng.dirichlet(np.ones(len(classes)))
        predicted = classes[int(np.argmax(proba))]
        actual = rng.choice(classes)
        snap = PredictionSnapshot(
            task_type="classification", model_name="m", model_version="1", feature_version="1",
            training_version="1", symbol="EUR_USD", timeframe="M5", prediction_horizon=5,
            timestamp="t", decision_index=i, feature_vector=[0.0], feature_names=["a"], confidence=60.0,
            predicted_class=predicted, class_probabilities=dict(zip(classes, proba.tolist())),
        )
        out.append(ResolvedPrediction(snapshot=snap, resolved_at="t", actual_class=actual))
    return out


def test_evaluate_needs_at_least_one_resolved() -> None:
    with pytest.raises(InsufficientDataError):
        PerformanceMonitor().evaluate([], "regression")


def test_unknown_task_type_raises() -> None:
    with pytest.raises(ValueError):
        PerformanceMonitor().evaluate(_regression_resolved(5), "bogus")


def test_classification_requires_classes() -> None:
    with pytest.raises(InsufficientDataError):
        PerformanceMonitor().evaluate(_classification_resolved(5, ["A", "B"]), "classification")


def test_regression_metrics_reuse_linear_regression_module() -> None:
    resolved = _regression_resolved(80)
    report = PerformanceMonitor(pip_size=0.0001).evaluate(resolved, "regression")
    for key in ("mae", "mse", "rmse", "r2", "mape", "explained_variance", "residual_statistics", "prediction_error_distribution"):
        assert key in report.metrics
    assert report.profit_factor is not None
    assert report.expected_vs_actual_pip_movement is not None
    assert report.n_samples == 80


def test_classification_metrics_reuse_logistic_regression_module() -> None:
    classes = ["SELL", "NO_TRADE", "BUY"]
    resolved = _classification_resolved(80, classes)
    report = PerformanceMonitor().evaluate(resolved, "classification", classes=classes)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss", "brier_score", "confusion_matrix"):
        assert key in report.metrics
    assert report.profit_factor is None
    assert report.expected_vs_actual_pip_movement is None


def test_profit_factor_high_for_consistently_correct_direction() -> None:
    rng = np.random.default_rng(2)
    resolved = []
    for i in range(50):
        pred = rng.uniform(0.0005, 0.002)
        actual = pred * rng.uniform(0.8, 1.2)  # always same sign, similar magnitude
        snap = PredictionSnapshot(
            task_type="regression", model_name="m", model_version="1", feature_version="1",
            training_version="1", symbol="EUR_USD", timeframe="M5", prediction_horizon=5,
            timestamp="t", decision_index=i, feature_vector=[0.0], feature_names=["a"], confidence=60.0,
            predicted_value=pred, raw_predictions={}, primary_target="next_return",
        )
        resolved.append(ResolvedPrediction(snapshot=snap, resolved_at="t", actual_value=actual))
    report = PerformanceMonitor().evaluate(resolved, "regression")
    assert report.profit_factor == float("inf") or report.profit_factor > 5.0


def test_rolling_vs_historical_windows() -> None:
    resolved = _regression_resolved(100)
    rh = PerformanceMonitor().rolling_vs_historical(resolved, "regression", window=20)
    assert rh.rolling.n_samples == 20
    assert rh.historical.n_samples == 100


def test_report_to_dict_serializable() -> None:
    resolved = _regression_resolved(30)
    report = PerformanceMonitor().evaluate(resolved, "regression")
    d = report.to_dict()
    assert d["task_type"] == "regression"
    assert d["n_samples"] == 30
