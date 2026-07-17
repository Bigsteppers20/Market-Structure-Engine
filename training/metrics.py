"""Reusable, extensible metric interfaces.

Every metric is a small :class:`Metric` implementation registered by name in
:data:`METRIC_REGISTRY`. New metrics -- for a future model type, a new risk
measure, whatever -- are added by calling :func:`register_metric` from
anywhere; nothing in this module needs to change (open/closed principle,
same pattern as ``ml_pipeline.label_generator``'s classification registry).

Metric *computation* here (MAE, accuracy, ...) is standard evaluation
arithmetic, not a predictive model -- it consumes ``y_true``/``y_pred``
arrays that some future trainer will supply; this module never fits
anything.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class Metric(ABC):
    """A single named metric, computed from true/predicted arrays."""

    name: ClassVar[str]
    task_type: ClassVar[str]  # "regression" | "classification"

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        raise NotImplementedError


class MAE(Metric):
    name, task_type = "mae", "regression"

    def compute(self, y_true, y_pred, **kwargs) -> float:
        return float(mean_absolute_error(y_true, y_pred))


class MSE(Metric):
    name, task_type = "mse", "regression"

    def compute(self, y_true, y_pred, **kwargs) -> float:
        return float(mean_squared_error(y_true, y_pred))


class RMSE(Metric):
    name, task_type = "rmse", "regression"

    def compute(self, y_true, y_pred, **kwargs) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class R2(Metric):
    name, task_type = "r2", "regression"

    def compute(self, y_true, y_pred, **kwargs) -> float:
        if len(y_true) < 2:
            return float("nan")
        return float(r2_score(y_true, y_pred))


class MAPE(Metric):
    name, task_type = "mape", "regression"

    def compute(self, y_true, y_pred, **kwargs) -> float:
        y_true = np.asarray(y_true, dtype=float)
        mask = y_true != 0
        if not mask.any():
            return float("nan")
        return float(mean_absolute_percentage_error(y_true[mask], np.asarray(y_pred)[mask]))


class Accuracy(Metric):
    name, task_type = "accuracy", "classification"

    def compute(self, y_true, y_pred, **kwargs) -> float:
        return float(accuracy_score(y_true, y_pred))


class Precision(Metric):
    name, task_type = "precision", "classification"

    def compute(self, y_true, y_pred, **kwargs) -> float:
        return float(precision_score(y_true, y_pred, average="macro", zero_division=0))


class Recall(Metric):
    name, task_type = "recall", "classification"

    def compute(self, y_true, y_pred, **kwargs) -> float:
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


class F1(Metric):
    name, task_type = "f1", "classification"

    def compute(self, y_true, y_pred, **kwargs) -> float:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


METRIC_REGISTRY: Dict[str, Metric] = {}


def register_metric(metric: Metric) -> None:
    """Register (or override) a metric by its ``name`` -- the sole extension point."""
    METRIC_REGISTRY[metric.name] = metric


for _m in (MAE(), MSE(), RMSE(), R2(), MAPE(), Accuracy(), Precision(), Recall(), F1()):
    register_metric(_m)


def metrics_for_task(task_type: str) -> List[str]:
    return [name for name, m in METRIC_REGISTRY.items() if m.task_type == task_type]


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute every registered regression metric."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {name: METRIC_REGISTRY[name].compute(y_true, y_pred) for name in metrics_for_task("regression")}


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Compute every registered classification metric, plus a confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: Dict[str, Any] = {
        name: METRIC_REGISTRY[name].compute(y_true, y_pred) for name in metrics_for_task("classification")
    }
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return out


# --------------------------------------------------------------------------- #
# Training / inference statistics -- distinct from prediction-quality metrics
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TrainingStatistics:
    """Statistics about the training run itself, not prediction quality."""

    duration_seconds: float
    n_train_samples: int
    n_val_samples: int
    n_test_samples: int
    n_features: int
    random_seed: int
    started_at: str
    finished_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "n_train_samples": self.n_train_samples,
            "n_val_samples": self.n_val_samples,
            "n_test_samples": self.n_test_samples,
            "n_features": self.n_features,
            "random_seed": self.random_seed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass(slots=True)
class InferenceStatistics:
    """Latency statistics for a batch of inference calls."""

    n_predictions: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    @classmethod
    def from_latencies(cls, latencies_ms: List[float]) -> "InferenceStatistics":
        if not latencies_ms:
            return cls(0, 0.0, 0.0, 0.0, 0.0)
        arr = np.asarray(latencies_ms, dtype=float)
        return cls(
            n_predictions=len(latencies_ms),
            mean_latency_ms=float(arr.mean()),
            p50_latency_ms=float(np.percentile(arr, 50)),
            p95_latency_ms=float(np.percentile(arr, 95)),
            p99_latency_ms=float(np.percentile(arr, 99)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_predictions": self.n_predictions,
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
        }
