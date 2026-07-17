"""Evaluation reporting for a trained classifier.

Wraps ``training.evaluator.EvaluationEngine`` (reused unmodified for
dataset/feature/training-statistics summaries and the confusion-matrix
placeholder-or-real logic) and adds every classification-specific
diagnostic the spec requires: calibration curves, coefficient magnitude/
stability, a full feature-importance report, and prediction/probability
distributions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from training.evaluator import EvaluationEngine
from training.metrics import TrainingStatistics

from .calibration import compute_calibration_curve
from .metrics import compute_all_classification_metrics


def build_feature_importance_report(coefficients: Optional[np.ndarray], feature_names: List[str]) -> Dict[str, Any]:
    """Standardized coefficients, positive/negative contributions, absolute
    importance, and top/bottom rankings -- every item the FEATURE IMPORTANCE
    spec section requires, computed from real fitted coefficients.

    "Standardized" because the model is always fit downstream of
    ``ml_pipeline.FeatureScaler`` -- the coefficients already operate on
    scaled inputs, no extra transform needed.
    """
    if coefficients is None:
        return {"status": "not available -- model has no coefficients (was it fit?)"}
    coef = np.atleast_2d(coefficients)
    signed_mean = coef.mean(axis=0)
    abs_importance = np.abs(coef).mean(axis=0)

    standardized = {name: float(v) for name, v in zip(feature_names, signed_mean)}
    absolute = {name: float(v) for name, v in zip(feature_names, abs_importance)}
    positive = dict(sorted(((n, v) for n, v in standardized.items() if v > 0), key=lambda kv: kv[1], reverse=True))
    negative = dict(sorted(((n, v) for n, v in standardized.items() if v < 0), key=lambda kv: kv[1]))
    ranked = sorted(absolute.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "standardized_coefficients": standardized,
        "positive_feature_contributions": positive,
        "negative_feature_contributions": negative,
        "absolute_feature_importance": absolute,
        "top_20_most_influential": ranked[:20],
        "least_influential": ranked[-10:] if len(ranked) >= 10 else ranked,
    }


def build_coefficient_diagnostics(
    coefficients: Optional[np.ndarray], bootstrap_coefficients: Optional[List[np.ndarray]], feature_names: List[str],
) -> Dict[str, Any]:
    """Coefficient magnitude (mean |coef| across class rows) and, if a
    bootstrap ensemble was fit, coefficient *stability* (std across the
    ensemble -- large std means the coefficient is unstable/not robust)."""
    if coefficients is None:
        return {"status": "not available"}
    coef = np.atleast_2d(coefficients)
    magnitude = {name: float(v) for name, v in zip(feature_names, np.abs(coef).mean(axis=0))}
    if not bootstrap_coefficients:
        return {"magnitude": magnitude, "stability": "not available -- n_bootstrap was 0"}
    stacked = np.stack([np.atleast_2d(c).mean(axis=0) for c in bootstrap_coefficients])
    stability = {name: float(v) for name, v in zip(feature_names, stacked.std(axis=0))}
    return {"magnitude": magnitude, "stability": stability}


def build_prediction_distribution(y_pred: Optional[np.ndarray], classes: List[str]) -> Dict[str, Any]:
    if y_pred is None:
        return {}
    values, counts = np.unique(y_pred, return_counts=True)
    total = int(counts.sum())
    dist = {classes[int(v)]: int(c) for v, c in zip(values, counts)}
    return {"counts": dist, "fractions": {k: v / total for k, v in dist.items()} if total else {}}


def build_probability_distribution(y_proba: Optional[np.ndarray], classes: List[str], bins: int = 10) -> Dict[str, Any]:
    if y_proba is None:
        return {}
    out: Dict[str, Any] = {}
    for i, cls in enumerate(classes):
        counts, edges = np.histogram(y_proba[:, i], bins=bins, range=(0.0, 1.0))
        out[cls] = {"bin_edges": edges.tolist(), "counts": counts.tolist()}
    return out


class ClassificationEvaluator:
    """Builds a richer classification evaluation report for one trained model."""

    def __init__(self) -> None:
        self._engine = EvaluationEngine()

    def evaluate(
        self,
        *,
        dataset_summary: Dict[str, Any],
        feature_names: List[str],
        classes: List[str],
        training_statistics: TrainingStatistics,
        y_val: Optional[np.ndarray] = None,
        y_val_pred: Optional[np.ndarray] = None,
        y_val_proba: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        y_test_pred: Optional[np.ndarray] = None,
        y_test_proba: Optional[np.ndarray] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        coefficients: Optional[np.ndarray] = None,
        bootstrap_coefficients: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        base_report = self._engine.evaluate(
            task_type="classification", dataset_summary=dataset_summary, feature_names=feature_names,
            training_statistics=training_statistics, y_val=y_val, y_val_pred=y_val_pred,
            y_test=y_test, y_test_pred=y_test_pred, y_test_proba=y_test_proba,
            feature_importance=feature_importance,
        )
        report = base_report.to_dict()

        if y_test is not None and y_test_pred is not None and y_test_proba is not None:
            report["testing_metrics"] = compute_all_classification_metrics(y_test, y_test_pred, y_test_proba)
        if y_val is not None and y_val_pred is not None and y_val_proba is not None:
            report["validation_metrics"] = compute_all_classification_metrics(y_val, y_val_pred, y_val_proba)

        if y_test is not None and y_test_proba is not None:
            report["calibration_curves"] = {
                cls: compute_calibration_curve((np.asarray(y_test) == i).astype(int), y_test_proba[:, i])
                for i, cls in enumerate(classes)
            }
        else:
            report["calibration_curves"] = {}

        report["coefficient_diagnostics"] = build_coefficient_diagnostics(coefficients, bootstrap_coefficients, feature_names)
        report["feature_importance_report"] = build_feature_importance_report(coefficients, feature_names)
        report["prediction_distribution"] = build_prediction_distribution(y_test_pred, classes)
        report["probability_distribution"] = build_probability_distribution(y_test_proba, classes)
        return report
