"""The Model Health Score: a single 0-100 number blending the 10 factors
the spec's MODEL HEALTH section requires.

Pure scoring function, same pattern as
``logistic_regression.confidence.compute_confidence``/
``linear_regression.confidence.compute_confidence``: it accepts already-
derived 0-100 (or neutral-default) factor scores, never a raw
``DriftReport``/``PerformanceReport``/lifecycle object itself -- deriving
those raw scores from the richer reports is ``health_engine.py``'s job.
Keeping this module a pure function makes every factor's contribution
independently testable without needing a full monitoring pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

#: The exact 10 factors the spec's MODEL HEALTH section names, and the
#: only valid keys for a ``weights`` override (``config.MonitorConfig``
#: validates against this same set).
HEALTH_SCORE_FACTORS = (
    "prediction_accuracy", "prediction_stability", "confidence_calibration",
    "feature_drift", "target_drift", "residual_drift", "rolling_error",
    "market_regime_change", "training_age", "prediction_coverage",
)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "prediction_accuracy": 0.20, "prediction_stability": 0.10, "confidence_calibration": 0.10,
    "feature_drift": 0.15, "target_drift": 0.08, "residual_drift": 0.08, "rolling_error": 0.10,
    "market_regime_change": 0.09, "training_age": 0.05, "prediction_coverage": 0.05,
}


@dataclass(slots=True)
class HealthScoreBreakdown:
    """The 10-factor health score, each 0-100, blended into ``.overall``."""

    prediction_accuracy: float
    prediction_stability: float
    confidence_calibration: float
    feature_drift: float
    target_drift: float
    residual_drift: float
    rolling_error: float
    market_regime_change: float
    training_age: float
    prediction_coverage: float
    weights: Dict[str, float]

    @property
    def overall(self) -> float:
        total = sum(getattr(self, k) * w for k, w in self.weights.items())
        return max(0.0, min(100.0, total))

    def to_dict(self) -> Dict[str, Any]:
        d = {k: round(getattr(self, k), 2) for k in HEALTH_SCORE_FACTORS}
        d["overall"] = round(self.overall, 2)
        return d


def severity_to_score(severity: float) -> float:
    """Map a [0, 1] drift/error *severity* to a 0-100 *health* contribution
    -- higher severity always means lower health."""
    return float(max(0.0, min(1.0, 1.0 - severity)) * 100.0)


def compute_health_score(
    *,
    prediction_accuracy: float,
    prediction_stability: float,
    confidence_calibration: float,
    feature_drift: float,
    rolling_error: float,
    prediction_coverage: float,
    target_drift: Optional[float] = None,
    residual_drift: Optional[float] = None,
    market_regime_change: Optional[float] = None,
    training_age: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> HealthScoreBreakdown:
    """Compute the 10-factor health score breakdown.

    Every argument is already a 0-100 score (higher = healthier) -- use
    :func:`severity_to_score` to convert a [0, 1] drift/error severity
    first. Optional factors default to a neutral 70 (not a penalty, not a
    reward) when the underlying diagnostic wasn't computable (e.g. no
    regime baseline configured yet, or the model was just trained so
    ``training_age`` doesn't yet apply).
    """
    w = weights or DEFAULT_WEIGHTS
    return HealthScoreBreakdown(
        prediction_accuracy=max(0.0, min(100.0, prediction_accuracy)),
        prediction_stability=max(0.0, min(100.0, prediction_stability)),
        confidence_calibration=max(0.0, min(100.0, confidence_calibration)),
        feature_drift=max(0.0, min(100.0, feature_drift)),
        target_drift=max(0.0, min(100.0, target_drift if target_drift is not None else 70.0)),
        residual_drift=max(0.0, min(100.0, residual_drift if residual_drift is not None else 70.0)),
        rolling_error=max(0.0, min(100.0, rolling_error)),
        market_regime_change=max(0.0, min(100.0, market_regime_change if market_regime_change is not None else 70.0)),
        training_age=max(0.0, min(100.0, training_age if training_age is not None else 70.0)),
        prediction_coverage=max(0.0, min(100.0, prediction_coverage)),
        weights=dict(w),
    )
