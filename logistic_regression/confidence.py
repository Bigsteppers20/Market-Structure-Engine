"""Prediction confidence, independent of the predicted *class*.

"Independent of the predicted class" means no class receives a systematic
confidence boost or penalty merely for being the one predicted -- every
factor below is symmetric across classes. It does *not* mean the confidence
score ignores the shape of the probability distribution: the spec itself
lists "class probability separation" as a required factor, and margin/
entropy-style measures of distribution shape are exactly that (they'd be
computed identically regardless of which class ends up on top).

Six factors, each 0-100, blended into one 0-100 ``ConfidenceBreakdown.overall``:

1. **Probability separation** -- the margin between the top-2 class
   probabilities (decisive vs. ambiguous).
2. **Historical accuracy** -- the model's balanced accuracy on an internal,
   held-out slice of training data.
3. **Distribution distance** -- how far the current (scaled) feature vector
   sits from the training distribution (mean |z-score| across features).
4. **Feature completeness** -- fraction of the current MarketState's
   ``_valid`` flags that are true.
5. **Prediction stability** -- agreement fraction of a bootstrap ensemble
   with the primary model's predicted class on this specific input.
6. **Calibration quality** -- how well the model's probabilities matched
   observed frequencies on held-out data (lower Brier/ECE = higher score).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

DEFAULT_WEIGHTS: Dict[str, float] = {
    "probability_separation": 0.20, "historical_accuracy": 0.20, "distribution_distance": 0.15,
    "feature_completeness": 0.15, "prediction_stability": 0.15, "calibration_quality": 0.15,
}


@dataclass(slots=True)
class ConfidenceBreakdown:
    probability_separation: float
    historical_accuracy: float
    distribution_distance: float
    feature_completeness: float
    prediction_stability: float
    calibration_quality: float
    weights: Dict[str, float]

    @property
    def overall(self) -> float:
        total = sum(getattr(self, k) * w for k, w in self.weights.items())
        return max(0.0, min(100.0, total))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probability_separation": round(self.probability_separation, 2),
            "historical_accuracy": round(self.historical_accuracy, 2),
            "distribution_distance": round(self.distribution_distance, 2),
            "feature_completeness": round(self.feature_completeness, 2),
            "prediction_stability": round(self.prediction_stability, 2),
            "calibration_quality": round(self.calibration_quality, 2),
            "overall": round(self.overall, 2),
        }


def _distribution_distance(mean_abs_z_score: float) -> float:
    return max(0.0, 100.0 - mean_abs_z_score * 20.0)


def _prediction_stability(ensemble_agreement: Optional[float]) -> float:
    if ensemble_agreement is None:
        return 50.0  # no bootstrap ensemble available -- neutral, not a penalty
    return max(0.0, min(1.0, ensemble_agreement)) * 100.0


def _calibration_quality(calibration_error: Optional[float]) -> float:
    if calibration_error is None:
        return 60.0  # unmeasured -- mildly neutral, not a penalty or a reward
    return max(0.0, 100.0 * (1.0 - min(calibration_error * 4.0, 1.0)))


def compute_confidence(
    *,
    margin: float,
    historical_balanced_accuracy: float,
    mean_abs_z_score: float,
    feature_completeness_fraction: float,
    ensemble_agreement: Optional[float] = None,
    calibration_error: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> ConfidenceBreakdown:
    """Compute the 6-factor confidence breakdown for one prediction.

    ``margin`` (from ``probability_engine.probability_margin``) is the only
    per-prediction-shaped input, and it is symmetric across classes -- no
    class is ever favored. Every other input is either a training-time
    statistic or derived purely from the input feature vector.
    """
    w = weights or DEFAULT_WEIGHTS
    return ConfidenceBreakdown(
        probability_separation=max(0.0, min(1.0, margin)) * 100.0,
        historical_accuracy=max(0.0, min(1.0, historical_balanced_accuracy)) * 100.0,
        distribution_distance=_distribution_distance(mean_abs_z_score),
        feature_completeness=max(0.0, min(1.0, feature_completeness_fraction)) * 100.0,
        prediction_stability=_prediction_stability(ensemble_agreement),
        calibration_quality=_calibration_quality(calibration_error),
        weights=dict(w),
    )
