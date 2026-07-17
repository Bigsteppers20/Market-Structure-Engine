"""Tests for logistic_regression.evaluator: model diagnostics (confusion
matrix via the wrapped training.EvaluationEngine, calibration curves,
coefficient magnitude/stability, feature importance report, prediction/
probability distributions)."""
from __future__ import annotations

import numpy as np
import pytest

from logistic_regression.evaluator import (
    ClassificationEvaluator,
    build_coefficient_diagnostics,
    build_feature_importance_report,
    build_prediction_distribution,
    build_probability_distribution,
)
from training.metrics import TrainingStatistics

FEATURE_NAMES = ["f0", "f1", "f2"]


def _stats() -> TrainingStatistics:
    return TrainingStatistics(
        duration_seconds=1.0, n_train_samples=100, n_val_samples=20, n_test_samples=20,
        n_features=3, random_seed=1, started_at="t0", finished_at="t1",
    )


# --------------------------------------------------------------------------- #
# Feature importance report
# --------------------------------------------------------------------------- #
def test_feature_importance_report_no_coefficients() -> None:
    report = build_feature_importance_report(None, FEATURE_NAMES)
    assert "status" in report


def test_feature_importance_report_splits_positive_negative() -> None:
    coef = np.array([[1.0, -2.0, 0.5], [0.5, -1.0, -0.5]])  # 2 one-vs-rest rows x 3 features
    report = build_feature_importance_report(coef, FEATURE_NAMES)
    assert report["standardized_coefficients"]["f0"] == pytest.approx(0.75)
    assert report["standardized_coefficients"]["f1"] == pytest.approx(-1.5)
    assert "f0" in report["positive_feature_contributions"]
    assert "f1" in report["negative_feature_contributions"]
    assert "f0" not in report["negative_feature_contributions"]
    assert set(report["absolute_feature_importance"]) == set(FEATURE_NAMES)
    assert len(report["top_20_most_influential"]) <= 20
    assert "least_influential" in report


def test_feature_importance_report_1d_coefficients() -> None:
    """Binary classification's sklearn coef_ can be shape (1, n_features) --
    must not crash on the mean-across-rows reduction."""
    coef = np.array([1.0, -2.0, 0.5])
    report = build_feature_importance_report(coef, FEATURE_NAMES)
    assert report["standardized_coefficients"]["f0"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Coefficient diagnostics
# --------------------------------------------------------------------------- #
def test_coefficient_diagnostics_no_coefficients() -> None:
    result = build_coefficient_diagnostics(None, None, FEATURE_NAMES)
    assert "status" in result


def test_coefficient_diagnostics_no_bootstrap() -> None:
    coef = np.array([[1.0, -2.0, 0.5]])
    result = build_coefficient_diagnostics(coef, None, FEATURE_NAMES)
    assert set(result["magnitude"]) == set(FEATURE_NAMES)
    assert result["stability"] == "not available -- n_bootstrap was 0"


def test_coefficient_diagnostics_with_bootstrap_stability() -> None:
    coef = np.array([[1.0, -2.0, 0.5]])
    bootstrap = [np.array([[1.1, -1.9, 0.4]]), np.array([[0.9, -2.1, 0.6]]), np.array([[1.0, -2.0, 0.5]])]
    result = build_coefficient_diagnostics(coef, bootstrap, FEATURE_NAMES)
    assert set(result["stability"]) == set(FEATURE_NAMES)
    assert all(v >= 0.0 for v in result["stability"].values())


# --------------------------------------------------------------------------- #
# Prediction / probability distributions
# --------------------------------------------------------------------------- #
def test_prediction_distribution_none_input() -> None:
    assert build_prediction_distribution(None, ["SELL", "NO_TRADE", "BUY"]) == {}


def test_prediction_distribution_counts_and_fractions() -> None:
    y_pred = np.array([0, 0, 1, 2, 2, 2])
    result = build_prediction_distribution(y_pred, ["SELL", "NO_TRADE", "BUY"])
    assert result["counts"] == {"SELL": 2, "NO_TRADE": 1, "BUY": 3}
    assert result["fractions"]["BUY"] == pytest.approx(0.5)


def test_probability_distribution_none_input() -> None:
    assert build_probability_distribution(None, ["SELL", "NO_TRADE", "BUY"]) == {}


def test_probability_distribution_has_one_histogram_per_class() -> None:
    rng = np.random.default_rng(0)
    y_proba = rng.random((50, 3))
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    result = build_probability_distribution(y_proba, ["SELL", "NO_TRADE", "BUY"], bins=5)
    assert set(result) == {"SELL", "NO_TRADE", "BUY"}
    for cls_hist in result.values():
        assert len(cls_hist["counts"]) == 5
        assert len(cls_hist["bin_edges"]) == 6


# --------------------------------------------------------------------------- #
# ClassificationEvaluator -- full report
# --------------------------------------------------------------------------- #
def _synthetic_test_split(n: int = 60, n_classes: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, n_classes, n)
    proba = np.zeros((n, n_classes))
    for i, cls in enumerate(y_true):
        proba[i, cls] = 0.7
        others = [c for c in range(n_classes) if c != cls]
        for o in others:
            proba[i, o] = 0.3 / len(others)
    y_pred = proba.argmax(axis=1)
    return y_true, y_pred, proba


def test_evaluator_full_report_contains_every_required_diagnostic() -> None:
    y_test, y_test_pred, y_test_proba = _synthetic_test_split()
    coef = np.array([[1.0, -2.0, 0.5], [0.2, 0.1, -0.3], [-1.0, 1.5, -0.2]])

    evaluator = ClassificationEvaluator()
    report = evaluator.evaluate(
        dataset_summary={"n_samples": 200, "target": "SELL_NO_TRADE_BUY"},
        feature_names=FEATURE_NAMES, classes=["SELL", "NO_TRADE", "BUY"],
        training_statistics=_stats(),
        y_test=y_test, y_test_pred=y_test_pred, y_test_proba=y_test_proba,
        feature_importance={"f0": 1.0, "f1": 1.5, "f2": 0.3},
        coefficients=coef,
    )
    assert "testing_metrics" in report
    assert "confusion_matrix" in report["testing_metrics"]
    assert "roc_auc" in report["testing_metrics"]
    assert "calibration_curves" in report
    assert set(report["calibration_curves"]) == {"SELL", "NO_TRADE", "BUY"}
    assert "coefficient_diagnostics" in report
    assert "feature_importance_report" in report
    assert "prediction_distribution" in report
    assert "probability_distribution" in report


def test_evaluator_handles_missing_test_split_gracefully() -> None:
    evaluator = ClassificationEvaluator()
    report = evaluator.evaluate(
        dataset_summary={"n_samples": 10, "target": "x"}, feature_names=FEATURE_NAMES,
        classes=["SELL", "NO_TRADE", "BUY"], training_statistics=_stats(),
    )
    assert report["calibration_curves"] == {}
    assert report["prediction_distribution"] == {}
    assert report["probability_distribution"] == {}
