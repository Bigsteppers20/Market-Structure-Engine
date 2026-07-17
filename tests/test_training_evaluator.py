"""Tests for training.evaluator."""
from __future__ import annotations

import numpy as np
import pytest

from training.evaluator import PLACEHOLDER, EvaluationEngine
from training.metrics import TrainingStatistics


def _stats() -> TrainingStatistics:
    return TrainingStatistics(
        duration_seconds=1.0, n_train_samples=100, n_val_samples=20, n_test_samples=20,
        n_features=185, random_seed=42, started_at="t0", finished_at="t1",
    )


def test_regression_report_with_predictions_is_real() -> None:
    y_val = np.array([1.0, 2.0, 3.0])
    y_test = np.array([1.0, 2.0, 3.0])
    report = EvaluationEngine().evaluate(
        task_type="regression", dataset_summary={"n_samples": 100},
        feature_names=["f1", "f2"], training_statistics=_stats(),
        y_val=y_val, y_val_pred=y_val, y_test=y_test, y_test_pred=np.array([1.1, 2.1, 2.9]),
    )
    assert report.validation_metrics["r2"] == pytest.approx(1.0)
    assert report.residual_analysis != PLACEHOLDER
    assert "mean" in report.residual_analysis
    assert report.confusion_matrix == PLACEHOLDER  # not a classification task
    assert report.calibration == PLACEHOLDER  # regression has no calibration
    assert report.feature_importance == PLACEHOLDER  # none supplied


def test_regression_report_without_predictions_is_placeholder() -> None:
    report = EvaluationEngine().evaluate(
        task_type="regression", dataset_summary={}, feature_names=["f1"],
        training_statistics=_stats(),
    )
    assert report.validation_metrics == PLACEHOLDER
    assert report.testing_metrics == PLACEHOLDER
    assert report.residual_analysis == PLACEHOLDER
    assert report.performance_summary == {"status": PLACEHOLDER}


def test_classification_report_with_predictions_is_real() -> None:
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    report = EvaluationEngine().evaluate(
        task_type="classification", dataset_summary={}, feature_names=["f1", "f2"],
        training_statistics=_stats(), y_val=y_test, y_val_pred=y_pred,
        y_test=y_test, y_test_pred=y_pred,
    )
    assert report.confusion_matrix != PLACEHOLDER
    cm = np.array(report.confusion_matrix)
    assert cm.sum() == 4
    assert report.residual_analysis == PLACEHOLDER  # not a regression task


def test_classification_report_with_probabilities_computes_calibration() -> None:
    rng = np.random.default_rng(0)
    y_test = rng.integers(0, 2, 60)
    proba = rng.uniform(0, 1, 60)
    y_pred = (proba > 0.5).astype(int)
    report = EvaluationEngine().evaluate(
        task_type="classification", dataset_summary={}, feature_names=["f1"],
        training_statistics=_stats(), y_test=y_test, y_test_pred=y_pred, y_test_proba=proba,
    )
    assert report.calibration != PLACEHOLDER
    assert "prob_true" in report.calibration


def test_feature_importance_passed_through_when_supplied() -> None:
    report = EvaluationEngine().evaluate(
        task_type="regression", dataset_summary={}, feature_names=["f1", "f2"],
        training_statistics=_stats(), feature_importance={"f1": 0.7, "f2": 0.3},
    )
    assert report.feature_importance == {"f1": 0.7, "f2": 0.3}


def test_invalid_task_type_raises() -> None:
    with pytest.raises(ValueError):
        EvaluationEngine().evaluate(
            task_type="bogus", dataset_summary={}, feature_names=[], training_statistics=_stats(),
        )


def test_report_to_dict_is_json_shaped() -> None:
    report = EvaluationEngine().evaluate(
        task_type="regression", dataset_summary={"n": 1}, feature_names=["f1"],
        training_statistics=_stats(),
    )
    d = report.to_dict()
    assert isinstance(d, dict)
    assert d["dataset_summary"] == {"n": 1}
