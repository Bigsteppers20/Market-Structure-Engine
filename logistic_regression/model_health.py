"""A single, per-model, training-time 0-100 "model health" score.

Distinct from ``confidence.py``'s per-prediction ``ConfidenceBreakdown``
(which varies per input): this is a model-level constant, computed once per
training run and attached to the fitted ``ClassificationModel`` -- the
classifier's counterpart to ``linear_regression/model_scoring.py``'s
``model_health_score`` idea, written independently here (never imported
from that sibling package).

Both inputs are already computed once per training run by
``trainer.py``'s internal holdout (``historical_balanced_accuracy_``,
``calibration_error_``) -- this is a free blend of existing numbers, no new
holdout or computation cost.
"""
from __future__ import annotations

from typing import Optional


def _accuracy_component(historical_balanced_accuracy: float) -> float:
    return max(0.0, min(1.0, historical_balanced_accuracy)) * 100.0


def _calibration_component(calibration_error: Optional[float]) -> float:
    if calibration_error is None:
        return 60.0  # unmeasured -- neutral, not a penalty (same convention as confidence.py)
    return max(0.0, 100.0 * (1.0 - min(calibration_error * 4.0, 1.0)))


def compute_model_health(historical_balanced_accuracy: float, calibration_error: Optional[float]) -> float:
    """Single 0-100 score blending historical (held-out) balanced accuracy
    (65% -- the primary signal of model quality) and calibration quality
    (35% -- secondary: a well-calibrated model is more trustworthy even at
    matched accuracy)."""
    accuracy = _accuracy_component(historical_balanced_accuracy)
    calibration = _calibration_component(calibration_error)
    return max(0.0, min(100.0, 0.65 * accuracy + 0.35 * calibration))
