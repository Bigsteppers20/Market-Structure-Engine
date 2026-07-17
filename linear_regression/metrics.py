"""Regression performance metrics.

Reuses ``training.metrics.compute_regression_metrics`` (MAE, MSE, RMSE, R²,
MAPE) unmodified, and extends it via the *exact* extension point documented
in ``TRAINING_INFRASTRUCTURE_REPORT.md``: register a new
``training.metrics.Metric`` for Explained Variance rather than
reimplementing metric aggregation. Residual statistics and the prediction
error distribution are genuinely new (not present in ``training.metrics``)
and are added here.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import explained_variance_score
from training.metrics import Metric, compute_regression_metrics, register_metric


class ExplainedVariance(Metric):
    name, task_type = "explained_variance", "regression"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        return float(explained_variance_score(y_true, y_pred))


register_metric(ExplainedVariance())


def residual_statistics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if residuals.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(residuals.mean()), "std": float(residuals.std(ddof=0)),
        "min": float(residuals.min()), "max": float(residuals.max()),
        "median": float(np.median(residuals)),
    }


def prediction_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> Dict[str, Any]:
    """Histogram of residuals -- the spec's "Prediction Error Distribution"."""
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if residuals.size == 0:
        return {"bin_edges": [], "counts": []}
    counts, edges = np.histogram(residuals, bins=bins)
    return {"bin_edges": edges.tolist(), "counts": counts.tolist()}


def compute_all_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """MAE/MSE/RMSE/R2/MAPE/explained_variance (via the shared training.metrics
    registry, now including ExplainedVariance) plus residual statistics and
    the prediction error distribution."""
    out = compute_regression_metrics(y_true, y_pred)
    out["residual_statistics"] = residual_statistics(y_true, y_pred)
    out["prediction_error_distribution"] = prediction_error_distribution(y_true, y_pred)
    return out
