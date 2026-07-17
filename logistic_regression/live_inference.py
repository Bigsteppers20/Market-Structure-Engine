"""The production LIVE INFERENCE contract.

Forex Dynamics does not want the Logistic Regression Engine's live serving
path to behave like an evaluation report -- no class probability
distribution, no confusion matrix, no accuracy/precision/recall/F1/ROC-AUC/
PR-AUC/log-loss/Brier score. Those stay exactly where they already lived
(training/evaluation reports, ``evaluator.py``, ``metrics.py`` -- all
untouched by this module). Live inference returns only the predicted
market direction and a confidence score suitable for the Decision Engine.

This module derives that minimal object from an already-computed
``predictor.ClassificationPrediction`` -- a pure function, no model/engine
access of its own, so it cannot change what the engine predicts or how
confidence is computed (``confidence.py`` stays the source of truth for
``prediction_confidence``). Everything richer (``class_probabilities``,
``probability_margin``, ``prediction_entropy``, ``confidence_breakdown``,
``explanation``) keeps flowing on the full ``ClassificationPrediction``
object itself -- this module never removes it, only omits it from the
minimal live view.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .predictor import ClassificationPrediction

#: The spec's three live-facing values. Only ``NO_TRADE`` needs remapping --
#: internal training/label/model class names stay ``SELL``/``NO_TRADE``/
#: ``BUY`` everywhere else (label_manager, model registry, evaluation
#: reports, the Decision Engine's own consumption of the full
#: ``ClassificationPrediction``) so nothing about how the model is trained,
#: registered, or evaluated changes -- only this one display-layer mapping.
#: Any class name not in this table (e.g. a caller-extended class set)
#: passes through unchanged; this minimal 3-value contract is scoped to the
#: engine's default ``(SELL, NO_TRADE, BUY)`` class set.
_LIVE_CLASS_MAP: Dict[str, str] = {"NO_TRADE": "WAIT"}


@dataclass(slots=True)
class LiveInferenceResponse:
    """The complete, minimal object a live prediction call returns."""

    prediction: str
    prediction_confidence: float
    prediction_horizon: int
    model_version: str
    feature_version: str
    training_version: str
    model_health: Optional[float]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "prediction_confidence": round(self.prediction_confidence),
            "prediction_horizon": self.prediction_horizon,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "training_version": self.training_version,
            "model_health": None if self.model_health is None else round(self.model_health),
            "timestamp": self.timestamp,
        }


def to_live_inference(prediction: ClassificationPrediction) -> LiveInferenceResponse:
    """Derive the minimal production contract from a full
    ``ClassificationPrediction`` -- the ONE place ``NO_TRADE`` becomes
    ``WAIT`` for display."""
    predicted = _LIVE_CLASS_MAP.get(prediction.predicted_class, prediction.predicted_class)
    return LiveInferenceResponse(
        prediction=predicted,
        prediction_confidence=prediction.prediction_confidence,
        prediction_horizon=prediction.prediction_horizon,
        model_version=prediction.model_version,
        feature_version=prediction.feature_version,
        training_version=prediction.training_version,
        model_health=prediction.model_health,
        timestamp=prediction.timestamp,
    )
