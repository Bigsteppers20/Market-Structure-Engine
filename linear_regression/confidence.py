"""Prediction confidence, independent of the predicted value itself.

Every factor below is derived from *training-time statistics* or the
*input feature vector* -- none of them ever reads the model's output. This
is enforced structurally: :func:`compute_confidence` doesn't even accept the
predicted value as a parameter (the new ``interval_width`` factor takes a
pre-computed ``interval_width_fraction`` ratio instead -- the caller divides
the interval width by the predicted value *before* calling this function,
so the predicted value itself never enters this module at all).

Seven factors, each 0-100, blended into one 0-100 ``ConfidenceBreakdown.overall``:

1. **Residual quality** -- how small the model's training residual std is,
   relative to the target's own scale.
2. **Historical accuracy** -- the model's held-out test-set R².
3. **Feature completeness** -- fraction of the current MarketState's
   ``_valid`` flags that are true (see ``feature_mapper.py``).
4. **Distribution distance** -- how far the current (scaled) feature vector
   sits from the training distribution (mean |z-score| across features).
5. **Prediction stability** -- how tightly a bootstrap ensemble of models
   agrees on this specific input (small ensemble spread = stable = confident).
6. **Cross-validation stability** -- walk-forward CV mean/std R² (see
   ``cross_validation.py``) -- neutral (60) when CV wasn't run for this
   model (``RegressionConfig.enable_cross_validation`` is opt-in, default
   off, so most models report this neutral default -- not a penalty).
7. **Interval width** -- the model's own prediction interval width,
   expressed as a fraction of the predicted value's own magnitude -- a
   distinct lens from prediction stability: two predictions can share the
   same absolute ensemble spread yet have very different *relative*
   precision if one predicts a much larger value than the other.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

DEFAULT_WEIGHTS: Dict[str, float] = {
    "residual_quality": 0.20, "historical_accuracy": 0.15, "feature_completeness": 0.15,
    "distribution_distance": 0.10, "prediction_stability": 0.15, "cv_stability": 0.15,
    "interval_width": 0.10,
}

_NEUTRAL_DEFAULT = 60.0


@dataclass(slots=True)
class ConfidenceBreakdown:
    residual_quality: float
    historical_accuracy: float
    feature_completeness: float
    distribution_distance: float
    prediction_stability: float
    weights: Dict[str, float]
    cv_stability: float = _NEUTRAL_DEFAULT
    interval_width: float = _NEUTRAL_DEFAULT

    @property
    def overall(self) -> float:
        total = sum(getattr(self, k) * w for k, w in self.weights.items())
        return max(0.0, min(100.0, total))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "residual_quality": round(self.residual_quality, 2),
            "historical_accuracy": round(self.historical_accuracy, 2),
            "feature_completeness": round(self.feature_completeness, 2),
            "distribution_distance": round(self.distribution_distance, 2),
            "prediction_stability": round(self.prediction_stability, 2),
            "cv_stability": round(self.cv_stability, 2),
            "interval_width": round(self.interval_width, 2),
            "overall": round(self.overall, 2),
        }


def _residual_quality(residual_std: float, target_std: float) -> float:
    if target_std <= 1e-12:
        return 50.0
    ratio = min(residual_std / target_std, 1.0)
    return max(0.0, 100.0 * (1.0 - ratio))


def _historical_accuracy(test_r2: float) -> float:
    return max(0.0, min(1.0, test_r2)) * 100.0


def _distribution_distance(mean_abs_z_score: float) -> float:
    return max(0.0, 100.0 - mean_abs_z_score * 20.0)


def _prediction_stability(ensemble_std: Optional[float], target_std: float) -> float:
    if ensemble_std is None:
        return 50.0  # no bootstrap ensemble available -- neutral, not a penalty
    if target_std <= 1e-12:
        return 50.0
    ratio = min(ensemble_std / target_std, 1.0)
    return max(0.0, 100.0 * (1.0 - ratio))


def _cv_stability(cv_mean_r2: Optional[float], cv_std_r2: Optional[float], reference_std: float = 0.5) -> float:
    if cv_mean_r2 is None or cv_std_r2 is None:
        return _NEUTRAL_DEFAULT  # cross-validation wasn't run for this model -- neutral, not a penalty
    mean_component = max(0.0, min(cv_mean_r2, 1.0)) * 100.0
    stability_component = max(0.0, 100.0 * (1.0 - min(cv_std_r2 / reference_std, 1.0)))
    return max(0.0, min(100.0, 0.6 * mean_component + 0.4 * stability_component))


def _interval_width(interval_width_fraction: Optional[float], reference_fraction: float = 0.5) -> float:
    if interval_width_fraction is None:
        return _NEUTRAL_DEFAULT  # no interval available (e.g. no bootstrap ensemble)
    return max(0.0, min(100.0, 100.0 * (1.0 - min(abs(interval_width_fraction) / reference_fraction, 1.0))))


def compute_confidence(
    *,
    residual_std: float,
    target_std: float,
    test_r2: float,
    feature_completeness_fraction: float,
    mean_abs_z_score: float,
    ensemble_std: Optional[float] = None,
    cv_mean_r2: Optional[float] = None,
    cv_std_r2: Optional[float] = None,
    interval_width_fraction: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> ConfidenceBreakdown:
    """Compute the 7-factor confidence breakdown for one prediction.

    Note the signature: no predicted value is accepted, only training-time
    statistics (``residual_std``, ``target_std``, ``test_r2``,
    ``cv_mean_r2``/``cv_std_r2``) and input-derived quantities
    (``feature_completeness_fraction``, ``mean_abs_z_score``,
    ``ensemble_std``, ``interval_width_fraction``) -- confidence cannot, by
    construction, depend on what the model predicted or whether it was
    *correct*. ``interval_width_fraction`` is the caller's own
    already-computed ``interval_width / |predicted_value|`` ratio -- this
    module never sees the predicted value itself, only that one derived
    ratio, preserving the same structural guarantee the original 5-factor
    design established (see ``test_lr_confidence.py``).
    """
    w = weights or DEFAULT_WEIGHTS
    return ConfidenceBreakdown(
        residual_quality=_residual_quality(residual_std, target_std),
        historical_accuracy=_historical_accuracy(test_r2),
        feature_completeness=max(0.0, min(1.0, feature_completeness_fraction)) * 100.0,
        distribution_distance=_distribution_distance(mean_abs_z_score),
        prediction_stability=_prediction_stability(ensemble_std, target_std),
        cv_stability=_cv_stability(cv_mean_r2, cv_std_r2),
        interval_width=_interval_width(interval_width_fraction),
        weights=dict(w),
    )
