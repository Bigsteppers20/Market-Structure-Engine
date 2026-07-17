"""Probability calibration: Platt Scaling, Isotonic Regression, or none.

Wraps scikit-learn's ``CalibratedClassifierCV`` around a ``FrozenEstimator``
-- the base estimator must already be fit on data *disjoint* from the
calibration set (see ``trainer.py``'s internal holdout), otherwise
calibration would overfit and report false confidence. ``FrozenEstimator``
is the modern (sklearn >= 1.6) replacement for the removed ``cv="prefit"``
mode: it marks an already-fitted estimator as "do not refit", letting
``CalibratedClassifierCV`` calibrate it directly on ``(X_cal, y_cal)``
instead of internally cross-validating a fresh fit.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

from .exceptions import CalibrationError

CALIBRATION_METHODS = ("none", "platt", "isotonic")

_SKLEARN_METHOD = {"platt": "sigmoid", "isotonic": "isotonic"}


def calibrate_estimator(base_estimator: Any, X_cal: np.ndarray, y_cal: np.ndarray, method: str) -> Tuple[Any, Dict[str, Any]]:
    """Wrap an already-fit ``base_estimator`` with a calibration layer.

    Returns ``(estimator_to_use, calibration_metadata)`` -- when
    ``method == "none"``, the base estimator is returned unchanged and the
    metadata records that explicitly.
    """
    if method not in CALIBRATION_METHODS:
        raise CalibrationError(f"Unknown calibration method {method!r}, expected one of {CALIBRATION_METHODS}.")
    if method == "none":
        return base_estimator, {"method": "none", "n_calibration_samples": 0}

    calibrated = CalibratedClassifierCV(FrozenEstimator(base_estimator), method=_SKLEARN_METHOD[method])
    calibrated.fit(X_cal, y_cal)
    return calibrated, {"method": method, "n_calibration_samples": int(len(X_cal))}


def compute_calibration_curve(y_true_binary: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Dict[str, List[float]]:
    """Reliability curve for one class (one-vs-rest binary labels)."""
    n_bins = max(2, min(n_bins, len(np.unique(y_proba))) if len(y_proba) else n_bins)
    try:
        prob_true, prob_pred = calibration_curve(y_true_binary, y_proba, n_bins=n_bins)
    except ValueError:
        return {"prob_true": [], "prob_pred": []}
    return {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}


def brier_score(y_true_binary: np.ndarray, y_proba: np.ndarray) -> float:
    return float(brier_score_loss(y_true_binary, y_proba))


def expected_calibration_error(y_true_binary: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Mean absolute gap between predicted probability and observed
    frequency, weighted by bin occupancy (standard ECE definition)."""
    y_true_binary = np.asarray(y_true_binary)
    y_proba = np.asarray(y_proba)
    if y_proba.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_proba, bin_edges[1:-1]), 0, n_bins - 1)
    total = len(y_proba)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        bin_confidence = y_proba[mask].mean()
        bin_accuracy = y_true_binary[mask].mean()
        ece += (mask.sum() / total) * abs(bin_confidence - bin_accuracy)
    return float(ece)
