"""Classification performance metrics and diagnostics.

Reuses ``training.metrics.compute_classification_metrics`` (accuracy,
precision, recall, F1, confusion matrix) unmodified, and extends it via the
same registry-extension pattern used across this platform
(``register_metric()``) with two label-only additions (Specificity,
Balanced Accuracy).

ROC-AUC, PR-AUC, Log Loss, and Brier Score need *probabilities*, not just
predicted labels -- ``training.metrics.compute_classification_metrics()``'s
generic dispatch loop (used by ``training.Trainer.run()`` for every
classification model on the platform) only ever calls
``Metric.compute(y_true, y_pred)``, with no ``y_proba``. These four are
still implemented as proper ``Metric`` subclasses (importable/usable
directly), but deliberately kept out of the shared ``METRIC_REGISTRY`` --
registering them there would make that generic call raise for every
classification trainer, not just this one. They are computed here via
:func:`compute_all_classification_metrics`, which supplies probabilities
directly.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.metrics import auc as sklearn_auc
from training.metrics import Metric, compute_classification_metrics, register_metric


class Specificity(Metric):
    """Macro-averaged true-negative rate across classes (one-vs-rest)."""

    name, task_type = "specificity", "classification"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        specificities = []
        for cls in classes:
            tn = np.sum((y_true != cls) & (y_pred != cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        return float(np.mean(specificities))


class BalancedAccuracy(Metric):
    name, task_type = "balanced_accuracy", "classification"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        return float(balanced_accuracy_score(y_true, y_pred))


class RocAuc(Metric):
    """Registered for discoverability; requires y_proba (see module docstring)."""

    name, task_type = "roc_auc", "classification"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        y_proba = kwargs.get("y_proba")
        if y_proba is None:
            raise ValueError("RocAuc.compute() requires y_proba=... (not available via the generic dispatcher).")
        return _roc_auc_macro(y_true, y_proba)


class PrAuc(Metric):
    name, task_type = "pr_auc", "classification"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        y_proba = kwargs.get("y_proba")
        if y_proba is None:
            raise ValueError("PrAuc.compute() requires y_proba=... (not available via the generic dispatcher).")
        return _pr_auc_macro(y_true, y_proba)


class LogLoss(Metric):
    name, task_type = "log_loss", "classification"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        y_proba = kwargs.get("y_proba")
        if y_proba is None:
            raise ValueError("LogLoss.compute() requires y_proba=... (not available via the generic dispatcher).")
        labels = kwargs.get("labels")
        return float(log_loss(y_true, y_proba, labels=labels))


class BrierScoreMetric(Metric):
    """Macro-averaged one-vs-rest Brier score."""

    name, task_type = "brier_score", "classification"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        y_proba = kwargs.get("y_proba")
        if y_proba is None:
            raise ValueError("BrierScoreMetric.compute() requires y_proba=... (not available via the generic dispatcher).")
        return _brier_macro(y_true, y_proba)


for _m in (Specificity(), BalancedAccuracy()):
    register_metric(_m)
#: RocAuc/PrAuc/LogLoss/BrierScoreMetric are deliberately NOT registered via
#: register_metric(): training.metrics.compute_classification_metrics() (used
#: by the unmodified training.Trainer.run() for every classification model,
#: including this one) iterates every *registered* classification metric
#: and calls it as compute(y_true, y_pred) -- no y_proba. Registering a
#: metric that requires y_proba would make that generic call raise for
#: every classification trainer on the platform, not just this engine's.
#: They stay as real Metric subclasses (usable directly, e.g. from a
#: notebook) and are computed here via compute_all_classification_metrics(),
#: which supplies y_proba explicitly.


def _roc_auc_macro(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    classes = np.arange(y_proba.shape[1])
    scores = []
    for i in classes:
        y_binary = (y_true == i).astype(int)
        if len(np.unique(y_binary)) < 2:
            continue
        scores.append(roc_auc_score(y_binary, y_proba[:, i]))
    return float(np.mean(scores)) if scores else 0.5


def _pr_auc_macro(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    scores = []
    for i in range(y_proba.shape[1]):
        y_binary = (y_true == i).astype(int)
        if len(np.unique(y_binary)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_binary, y_proba[:, i])
        scores.append(sklearn_auc(recall, precision))
    return float(np.mean(scores)) if scores else 0.0


def _brier_macro(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    scores = []
    for i in range(y_proba.shape[1]):
        y_binary = (y_true == i).astype(int)
        scores.append(brier_score_loss(y_binary, y_proba[:, i]))
    return float(np.mean(scores))


def compute_all_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, labels: List[Any] | None = None,
) -> Dict[str, Any]:
    """accuracy/precision/recall/f1/confusion_matrix (via the shared
    training.metrics registry) + specificity/balanced_accuracy (also via
    the registry) + the 4 probability-dependent metrics computed directly."""
    out = compute_classification_metrics(y_true, y_pred, labels=labels)
    out["roc_auc"] = _roc_auc_macro(y_true, y_proba)
    out["pr_auc"] = _pr_auc_macro(y_true, y_proba)
    out["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
    out["brier_score"] = _brier_macro(y_true, y_proba)
    return out
