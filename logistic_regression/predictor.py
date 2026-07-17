"""Turns a live ``MarketState`` into a ``ClassificationPrediction``.

``feature_mapper`` extracts the input vector, ``ClassificationInferencePipeline``
produces a probability vector + bootstrap agreement, ``confidence`` turns
training-time statistics + the input vector into a 0-100 score (never
favoring the predicted class), ``probability_engine`` derives the predicted
class/margin/entropy, and this module assembles the result plus a
deterministic, structured explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from market_structure import MarketState
from training.utils import utc_timestamp

from .classification_model import ClassificationModel
from .confidence import ConfidenceBreakdown, compute_confidence
from .feature_mapper import extract_feature_vector, feature_completeness
from .inference import ClassificationInferencePipeline
from .probability_engine import (
    assert_probabilities_sum_to_one,
    predicted_class,
    prediction_entropy,
    probability_margin,
    to_class_probabilities,
)


@dataclass(slots=True)
class ClassificationPrediction:
    """Output of one ``ClassificationPredictor.predict()`` call."""

    buy_probability: Optional[float]
    sell_probability: Optional[float]
    no_trade_probability: Optional[float]
    predicted_class: str
    prediction_confidence: float
    probability_margin: float
    prediction_entropy: float
    model_version: str
    feature_version: str
    training_version: str
    timestamp: str
    symbol: str
    timeframe: str
    prediction_horizon: int
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    explanation: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # --- Decision Engine metadata (additive; a per-model training-time
    # constant, distinct from prediction_confidence which varies per input --
    # see model_health.py). ---
    model_health: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buy_probability": self.buy_probability, "sell_probability": self.sell_probability,
            "no_trade_probability": self.no_trade_probability, "predicted_class": self.predicted_class,
            "prediction_confidence": round(self.prediction_confidence, 2),
            "probability_margin": round(self.probability_margin, 4),
            "prediction_entropy": round(self.prediction_entropy, 4),
            "model_version": self.model_version, "feature_version": self.feature_version,
            "training_version": self.training_version, "timestamp": self.timestamp,
            "symbol": self.symbol, "timeframe": self.timeframe, "prediction_horizon": self.prediction_horizon,
            "class_probabilities": {k: round(v, 6) for k, v in self.class_probabilities.items()},
            "confidence_breakdown": self.confidence_breakdown,
            "explanation": self.explanation, "warnings": self.warnings,
            "model_health": None if self.model_health is None else round(self.model_health, 2),
        }


class ClassificationPredictor:
    """Wraps one loaded :class:`ClassificationInferencePipeline`."""

    def __init__(self, pipeline: ClassificationInferencePipeline, top_n_features: int = 5) -> None:
        if pipeline.model is None or pipeline.model_version is None:
            raise ValueError("ClassificationPredictor requires an already-loaded pipeline.")
        self.pipeline = pipeline
        self.top_n_features = top_n_features

    def predict(self, market_state: MarketState, symbol: str = "UNKNOWN", timeframe: str = "UNKNOWN") -> ClassificationPrediction:
        X, feature_names = extract_feature_vector(market_state)
        completeness = feature_completeness(market_state)
        warnings: List[str] = []

        proba_row, agreement = self.pipeline.predict_proba(X, feature_names)
        model = self.pipeline.model
        assert model is not None and self.pipeline.model_version is not None

        class_probs = to_class_probabilities(proba_row.ravel(), model.classes_)
        assert_probabilities_sum_to_one(class_probs)

        predicted = predicted_class(class_probs)
        margin = probability_margin(class_probs)
        entropy = prediction_entropy(class_probs)

        z_scores = np.abs((X.ravel() - model.train_feature_mean_) / model.train_feature_std_)
        ensemble_agreement = float(np.ravel(agreement)[0]) if agreement is not None else None

        breakdown = compute_confidence(
            margin=margin, historical_balanced_accuracy=model.historical_balanced_accuracy_,
            mean_abs_z_score=float(np.mean(z_scores)), feature_completeness_fraction=completeness,
            ensemble_agreement=ensemble_agreement, calibration_error=model.calibration_error_,
        )

        explanation = _generate_explanation(
            predicted=predicted, class_probs=class_probs, confidence=breakdown.overall, margin=margin,
            historical_accuracy=model.historical_balanced_accuracy_,
            calibration_method=model.calibration_metadata_.get("method", "none"),
            model=model, feature_names=feature_names, top_n=self.top_n_features,
        )

        version = self.pipeline.model_version
        return ClassificationPrediction(
            buy_probability=class_probs.get("BUY"), sell_probability=class_probs.get("SELL"),
            no_trade_probability=class_probs.get("NO_TRADE"), predicted_class=predicted,
            prediction_confidence=breakdown.overall, probability_margin=margin, prediction_entropy=entropy,
            model_version=version.model_version, feature_version=version.version_info.feature_version,
            training_version=version.version_info.training_pipeline_version, timestamp=utc_timestamp(),
            symbol=symbol, timeframe=timeframe, prediction_horizon=version.prediction_horizon,
            class_probabilities=class_probs, confidence_breakdown=breakdown.to_dict(),
            explanation=explanation, warnings=warnings,
            model_health=getattr(model, "model_health_", None),
        )


def _generate_explanation(
    *, predicted: str, class_probs: Dict[str, float], confidence: float, margin: float,
    historical_accuracy: float, calibration_method: str, model: ClassificationModel,
    feature_names: List[str], top_n: int,
) -> List[str]:
    lines = [f"Predicted class: {predicted}"]
    for cls, p in class_probs.items():
        lines.append(f"{cls} probability: {p * 100:.1f}%")
    lines.append(f"Prediction confidence: {confidence:.0f}%")
    lines.append(f"Probability margin: {margin * 100:.1f}%")
    lines.append(f"Historical accuracy: {historical_accuracy * 100:.1f}%")
    lines.append(f"Calibration: {calibration_method}")

    coef = model.coefficients
    if coef is not None:
        signed = np.atleast_2d(coef).mean(axis=0)
        ranked = sorted(zip(feature_names, signed), key=lambda kv: kv[1], reverse=True)
        top_positive = [name for name, v in ranked if v > 0][:top_n]
        top_negative = [name for name, v in reversed(ranked) if v < 0][:top_n]
        if top_positive:
            lines.append(f"Top positive features: {', '.join(top_positive)}")
        if top_negative:
            lines.append(f"Top negative features: {', '.join(top_negative)}")
        influential = sorted(zip(feature_names, np.abs(signed)), key=lambda kv: kv[1], reverse=True)[:top_n]
        lines.append(f"Most influential market structure signals: {', '.join(n for n, _ in influential)}")
    return lines
