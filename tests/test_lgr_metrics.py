"""Tests for logistic_regression.metrics: registry extensions (Specificity,
BalancedAccuracy, RocAuc, PrAuc, LogLoss, BrierScoreMetric) and the
compute_all_classification_metrics() aggregate used by the evaluator."""
from __future__ import annotations

import numpy as np
import pytest

from logistic_regression.metrics import (
    BalancedAccuracy,
    BrierScoreMetric,
    LogLoss,
    PrAuc,
    RocAuc,
    Specificity,
    compute_all_classification_metrics,
)
from training.metrics import METRIC_REGISTRY, compute_classification_metrics


def _perfect_binary_case():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.05, 0.95]])
    return y_true, y_pred, y_proba


def test_label_only_metrics_registered_in_shared_registry() -> None:
    """Specificity/BalancedAccuracy work from (y_true, y_pred) alone, so they
    are safe to register in the shared training.metrics.METRIC_REGISTRY."""
    for name in ("specificity", "balanced_accuracy"):
        assert name in METRIC_REGISTRY
        assert METRIC_REGISTRY[name].task_type == "classification"


def test_probability_dependent_metrics_not_in_shared_registry() -> None:
    """RocAuc/PrAuc/LogLoss/BrierScoreMetric require y_proba, which
    training.metrics.compute_classification_metrics()'s generic dispatch
    loop (used by every classification Trainer.run() on the platform) never
    supplies -- registering them there would break every classification
    trainer, not just this one. They must stay out of METRIC_REGISTRY and be
    computed via compute_all_classification_metrics() instead."""
    for name in ("roc_auc", "pr_auc", "log_loss", "brier_score"):
        assert name not in METRIC_REGISTRY


def test_generic_classification_dispatch_still_works_after_import() -> None:
    """Regression guard: importing logistic_regression.metrics must not
    break training.metrics.compute_classification_metrics() for ANY
    classification model on the platform (not just this engine's)."""
    y_true, y_pred, _ = _perfect_binary_case()
    out = compute_classification_metrics(y_true, y_pred)
    assert out["accuracy"] == pytest.approx(1.0)


def test_specificity_perfect_predictions() -> None:
    y_true, y_pred, _ = _perfect_binary_case()
    assert Specificity().compute(y_true, y_pred) == pytest.approx(1.0)


def test_balanced_accuracy_perfect_predictions() -> None:
    y_true, y_pred, _ = _perfect_binary_case()
    assert BalancedAccuracy().compute(y_true, y_pred) == pytest.approx(1.0)


def test_roc_auc_requires_y_proba() -> None:
    y_true, y_pred, _ = _perfect_binary_case()
    with pytest.raises(ValueError):
        RocAuc().compute(y_true, y_pred)


def test_roc_auc_perfect_separation() -> None:
    y_true, y_pred, y_proba = _perfect_binary_case()
    auc = RocAuc().compute(y_true, y_pred, y_proba=y_proba)
    assert auc == pytest.approx(1.0)


def test_pr_auc_requires_y_proba() -> None:
    y_true, y_pred, _ = _perfect_binary_case()
    with pytest.raises(ValueError):
        PrAuc().compute(y_true, y_pred)


def test_log_loss_requires_y_proba() -> None:
    y_true, y_pred, _ = _perfect_binary_case()
    with pytest.raises(ValueError):
        LogLoss().compute(y_true, y_pred)


def test_log_loss_confident_correct_is_low() -> None:
    y_true, y_pred, y_proba = _perfect_binary_case()
    loss = LogLoss().compute(y_true, y_pred, y_proba=y_proba, labels=[0, 1])
    assert loss < 0.3


def test_brier_score_metric_requires_y_proba() -> None:
    y_true, y_pred, _ = _perfect_binary_case()
    with pytest.raises(ValueError):
        BrierScoreMetric().compute(y_true, y_pred)


def test_brier_score_metric_low_for_confident_correct_predictions() -> None:
    y_true, y_pred, y_proba = _perfect_binary_case()
    score = BrierScoreMetric().compute(y_true, y_pred, y_proba=y_proba)
    assert score < 0.1


def test_compute_all_classification_metrics_includes_base_and_extension_metrics() -> None:
    y_true, y_pred, y_proba = _perfect_binary_case()
    out = compute_all_classification_metrics(y_true, y_pred, y_proba, labels=[0, 1])
    # From training.metrics (reused, not duplicated):
    for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
        assert key in out
    # From this module's probability-dependent additions:
    for key in ("roc_auc", "pr_auc", "log_loss", "brier_score"):
        assert key in out
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["roc_auc"] == pytest.approx(1.0)


def test_multiclass_roc_pr_auc_macro_average() -> None:
    rng = np.random.default_rng(2)
    n = 90
    y_true = rng.integers(0, 3, n)
    y_proba = np.zeros((n, 3))
    for i, cls in enumerate(y_true):
        y_proba[i, cls] = 0.8
        others = [c for c in range(3) if c != cls]
        y_proba[i, others] = 0.1
    y_pred = y_proba.argmax(axis=1)
    out = compute_all_classification_metrics(y_true, y_pred, y_proba)
    assert 0.0 <= out["roc_auc"] <= 1.0
    assert 0.0 <= out["pr_auc"] <= 1.0
    assert out["roc_auc"] > 0.8  # strongly separated synthetic probabilities


def test_base_classification_metrics_still_reused_unmodified() -> None:
    """compute_classification_metrics from training.metrics must remain the
    single source of truth for accuracy/precision/recall/f1/confusion_matrix
    -- this engine only adds probability-dependent metrics on top."""
    y_true, y_pred, _ = _perfect_binary_case()
    base = compute_classification_metrics(y_true, y_pred)
    assert "accuracy" in base and "confusion_matrix" in base
