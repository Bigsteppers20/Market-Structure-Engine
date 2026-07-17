"""Performance monitoring (PERFORMANCE MONITORING section).

Reuses ``linear_regression.metrics.compute_all_regression_metrics`` (MAE,
RMSE, MAPE, R², explained variance, residual statistics, prediction error
distribution) and ``logistic_regression.metrics.compute_all_classification_metrics``
(accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss, Brier score,
confusion matrix) unmodified -- this module never recomputes a metric
either of those two already compute correctly. Unlike ``linear_regression``/
``logistic_regression`` themselves (which must stay independent siblings),
``model_monitor`` sits *above* both by design -- its entire job is to
monitor them -- so importing both here is expected and intentional.

Profit factor and expected-vs-actual pip movement are regression-only
(they assume a pip/return-denominated target); classification's outcome is
categorical, so those two fields are ``None`` there and the richer
probability-based classification metrics fill that role instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .exceptions import InsufficientDataError
from .prediction_monitor import ResolvedPrediction


def _profit_factor(predicted: np.ndarray, actual: np.ndarray) -> Optional[float]:
    """Classic trading diagnostic: bet ``sign(predicted)`` size 1, realize
    ``actual`` -- ``sum(gains) / sum(|losses|)``. ``None`` if there's no
    signal to evaluate (every prediction was exactly zero)."""
    direction = np.sign(predicted)
    pnl = direction * actual
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if gains == 0 and losses == 0:
        return None
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def _expected_vs_actual_pip(resolved: Sequence[ResolvedPrediction], pip_size: float) -> Dict[str, float]:
    predicted = np.array([r.snapshot.predicted_value for r in resolved], dtype=float)
    actual = np.array([r.actual_value for r in resolved], dtype=float)
    predicted_pips = predicted / pip_size
    actual_pips = actual / pip_size
    correlation = float(np.corrcoef(predicted_pips, actual_pips)[0, 1]) if len(resolved) >= 2 else 0.0
    return {
        "expected_mean_pips": float(predicted_pips.mean()), "actual_mean_pips": float(actual_pips.mean()),
        "expected_std_pips": float(predicted_pips.std()), "actual_std_pips": float(actual_pips.std()),
        "correlation": 0.0 if np.isnan(correlation) else correlation,
    }


@dataclass(slots=True)
class PerformanceReport:
    """Full performance assessment for one batch of resolved predictions."""

    task_type: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    profit_factor: Optional[float] = None
    expected_vs_actual_pip_movement: Optional[Dict[str, float]] = None
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type, "metrics": self.metrics, "profit_factor": self.profit_factor,
            "expected_vs_actual_pip_movement": self.expected_vs_actual_pip_movement, "n_samples": self.n_samples,
        }


@dataclass(slots=True)
class RollingHistoricalPerformance:
    """Rolling (most-recent-window) vs. historical (all-time) performance."""

    rolling: PerformanceReport
    historical: PerformanceReport

    def to_dict(self) -> Dict[str, Any]:
        return {"rolling": self.rolling.to_dict(), "historical": self.historical.to_dict()}


class PerformanceMonitor:
    """Computes regression or classification performance from resolved predictions."""

    def __init__(self, pip_size: float = 0.0001) -> None:
        self.pip_size = pip_size

    def evaluate(
        self, resolved: Sequence[ResolvedPrediction], task_type: str, classes: Optional[List[str]] = None,
    ) -> PerformanceReport:
        if not resolved:
            raise InsufficientDataError("PerformanceMonitor.evaluate() needs >= 1 resolved prediction.")
        if task_type == "regression":
            return self._evaluate_regression(resolved)
        if task_type == "classification":
            if not classes:
                raise InsufficientDataError("classes= is required to evaluate a classification model.")
            return self._evaluate_classification(resolved, classes)
        raise ValueError(f"Unknown task_type {task_type!r} (must be 'regression' or 'classification').")

    def _evaluate_regression(self, resolved: Sequence[ResolvedPrediction]) -> PerformanceReport:
        from linear_regression.metrics import compute_all_regression_metrics

        usable = [r for r in resolved if r.actual_value is not None and r.snapshot.predicted_value is not None]
        if not usable:
            raise InsufficientDataError("No resolved regression prediction has both an actual and a predicted value.")
        y_true = np.array([r.actual_value for r in usable], dtype=float)
        y_pred = np.array([r.snapshot.predicted_value for r in usable], dtype=float)
        metrics = compute_all_regression_metrics(y_true, y_pred)
        return PerformanceReport(
            task_type="regression", metrics=metrics, profit_factor=_profit_factor(y_pred, y_true),
            expected_vs_actual_pip_movement=_expected_vs_actual_pip(usable, self.pip_size), n_samples=len(usable),
        )

    def _evaluate_classification(self, resolved: Sequence[ResolvedPrediction], classes: List[str]) -> PerformanceReport:
        from logistic_regression.metrics import compute_all_classification_metrics

        usable = [
            r for r in resolved
            if r.actual_class is not None and r.snapshot.predicted_class is not None and r.snapshot.class_probabilities
        ]
        if not usable:
            raise InsufficientDataError("No resolved classification prediction has both an actual class and probabilities.")
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_true = np.array([class_to_idx[r.actual_class] for r in usable if r.actual_class in class_to_idx])
        y_pred = np.array([class_to_idx[r.snapshot.predicted_class] for r in usable if r.actual_class in class_to_idx])
        y_proba = np.array([
            [r.snapshot.class_probabilities.get(c, 0.0) for c in classes]
            for r in usable if r.actual_class in class_to_idx
        ])
        metrics = compute_all_classification_metrics(y_true, y_pred, y_proba, labels=list(range(len(classes))))
        return PerformanceReport(
            task_type="classification", metrics=metrics, profit_factor=None,
            expected_vs_actual_pip_movement=None, n_samples=len(usable),
        )

    def rolling_vs_historical(
        self, resolved: Sequence[ResolvedPrediction], task_type: str, window: int, classes: Optional[List[str]] = None,
    ) -> RollingHistoricalPerformance:
        historical = self.evaluate(resolved, task_type, classes=classes)
        rolling = self.evaluate(list(resolved)[-window:], task_type, classes=classes)
        return RollingHistoricalPerformance(rolling=rolling, historical=historical)
