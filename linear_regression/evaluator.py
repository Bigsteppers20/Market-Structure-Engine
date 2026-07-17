"""Evaluation reporting for a trained regression target.

Thin wrapper around ``training.evaluator.EvaluationEngine`` (reused
unmodified for dataset/feature/training-statistics summaries and the
residual-analysis placeholder-or-real logic) plus this engine's own
richer regression metrics (``metrics.compute_all_regression_metrics``,
which already includes residual statistics and the error distribution).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from training.evaluator import EvaluationEngine, EvaluationReport
from training.metrics import TrainingStatistics

from .metrics import compute_all_regression_metrics


class RegressionEvaluator:
    """Builds a richer regression evaluation report for one target."""

    def __init__(self) -> None:
        self._engine = EvaluationEngine()

    def evaluate(
        self,
        *,
        dataset_summary: Dict[str, Any],
        feature_names: list[str],
        training_statistics: TrainingStatistics,
        y_val: Optional[np.ndarray] = None,
        y_val_pred: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        y_test_pred: Optional[np.ndarray] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        base_report: EvaluationReport = self._engine.evaluate(
            task_type="regression", dataset_summary=dataset_summary, feature_names=feature_names,
            training_statistics=training_statistics, y_val=y_val, y_val_pred=y_val_pred,
            y_test=y_test, y_test_pred=y_test_pred, feature_importance=feature_importance,
        )
        report = base_report.to_dict()
        if y_test is not None and y_test_pred is not None:
            report["testing_metrics"] = compute_all_regression_metrics(y_test, y_test_pred)
        if y_val is not None and y_val_pred is not None:
            report["validation_metrics"] = compute_all_regression_metrics(y_val, y_val_pred)
        return report
