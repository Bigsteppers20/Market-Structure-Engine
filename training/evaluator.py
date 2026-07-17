"""Evaluation report generation.

Produces real diagnostics (confusion matrix, residual analysis, calibration,
feature importance) whenever the caller supplies the data they need
(predictions, probabilities, an importance mapping) -- and an explicit,
clearly-labeled placeholder otherwise. Since this package implements no
model, every placeholder path is exercised today; a future concrete
:class:`training.trainer.Trainer` subclass lights up the real path just by
supplying more data, with zero changes to this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from .metrics import TrainingStatistics, compute_classification_metrics, compute_regression_metrics

PLACEHOLDER = "not available -- no predictions/probabilities/importances were supplied"


@dataclass(slots=True)
class EvaluationReport:
    dataset_summary: Dict[str, Any]
    feature_summary: Dict[str, Any]
    training_statistics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    testing_metrics: Dict[str, Any]
    performance_summary: Dict[str, Any]
    confusion_matrix: Any
    residual_analysis: Any
    calibration: Any
    feature_importance: Any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationEngine:
    """Builds an :class:`EvaluationReport` from whatever a trainer provides."""

    def evaluate(
        self,
        *,
        task_type: str,
        dataset_summary: Dict[str, Any],
        feature_names: List[str],
        training_statistics: TrainingStatistics,
        y_val: Optional[np.ndarray] = None,
        y_val_pred: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        y_test_pred: Optional[np.ndarray] = None,
        y_test_proba: Optional[np.ndarray] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> EvaluationReport:
        if task_type not in ("regression", "classification"):
            raise ValueError("task_type must be 'regression' or 'classification'.")

        compute_fn = compute_regression_metrics if task_type == "regression" else compute_classification_metrics
        val_metrics = compute_fn(y_val, y_val_pred) if y_val is not None and y_val_pred is not None else PLACEHOLDER
        test_metrics = compute_fn(y_test, y_test_pred) if y_test is not None and y_test_pred is not None else PLACEHOLDER

        return EvaluationReport(
            dataset_summary=dataset_summary,
            feature_summary={
                "feature_count": len(feature_names),
                "feature_names_sample": feature_names[:10],
            },
            training_statistics=training_statistics.to_dict(),
            validation_metrics=val_metrics,
            testing_metrics=test_metrics,
            performance_summary=self._performance_summary(task_type, val_metrics, test_metrics),
            confusion_matrix=self._confusion_matrix(task_type, y_test, y_test_pred),
            residual_analysis=self._residual_analysis(task_type, y_test, y_test_pred),
            calibration=self._calibration(task_type, y_test, y_test_proba),
            feature_importance=feature_importance if feature_importance else PLACEHOLDER,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _performance_summary(task_type: str, val_metrics: Any, test_metrics: Any) -> Dict[str, Any]:
        if val_metrics == PLACEHOLDER or test_metrics == PLACEHOLDER:
            return {"status": PLACEHOLDER}
        primary = "r2" if task_type == "regression" else "accuracy"
        return {
            "primary_metric": primary,
            "validation": val_metrics.get(primary),
            "testing": test_metrics.get(primary),
            "generalization_gap": (
                (val_metrics.get(primary) - test_metrics.get(primary))
                if val_metrics.get(primary) is not None and test_metrics.get(primary) is not None
                else None
            ),
        }

    @staticmethod
    def _confusion_matrix(task_type: str, y_test, y_test_pred) -> Any:
        if task_type != "classification" or y_test is None or y_test_pred is None:
            return PLACEHOLDER
        return confusion_matrix(y_test, y_test_pred).tolist()

    @staticmethod
    def _residual_analysis(task_type: str, y_test, y_test_pred) -> Any:
        if task_type != "regression" or y_test is None or y_test_pred is None:
            return PLACEHOLDER
        residuals = np.asarray(y_test, dtype=float) - np.asarray(y_test_pred, dtype=float)
        return {
            "mean": float(residuals.mean()),
            "std": float(residuals.std(ddof=0)),
            "min": float(residuals.min()),
            "max": float(residuals.max()),
            "median": float(np.median(residuals)),
        }

    @staticmethod
    def _calibration(task_type: str, y_test, y_test_proba) -> Any:
        if task_type != "classification" or y_test is None or y_test_proba is None:
            return PLACEHOLDER
        y_test_arr = np.asarray(y_test)
        proba_arr = np.asarray(y_test_proba)
        if proba_arr.ndim == 2:
            # Binary-style calibration against the positive-class column.
            if proba_arr.shape[1] != 2:
                return PLACEHOLDER
            proba_arr = proba_arr[:, 1]
        classes = np.unique(y_test_arr)
        if len(classes) != 2:
            return PLACEHOLDER
        pos_label = classes[-1]
        y_binary = (y_test_arr == pos_label).astype(int)
        prob_true, prob_pred = calibration_curve(y_binary, proba_arr, n_bins=min(10, len(y_binary)))
        return {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist(), "positive_label": str(pos_label)}
