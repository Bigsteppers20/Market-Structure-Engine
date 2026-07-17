"""Turns a live ``MarketState`` into a ``RegressionPrediction``.

This is the only place the engine's pieces come together for inference:
``feature_mapper`` extracts the input vector, one ``RegressionInferencePipeline``
per configured target produces a point estimate + uncertainty, ``confidence``
turns training-time statistics + the input vector into a 0-100 score
(never the predicted value), and this module assembles the result plus a
structured, human-readable explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from market_structure import MarketState
from training.utils import utc_timestamp

from .confidence import ConfidenceBreakdown, compute_confidence
from .feature_mapper import extract_feature_vector, feature_completeness
from .inference import RegressionInferencePipeline
from .target_generator import TARGET_TO_PREDICTION_FIELD

#: Fields the spec's OUTPUT OBJECT names explicitly.
_NAMED_FIELDS = tuple(TARGET_TO_PREDICTION_FIELD.values())


@dataclass(slots=True)
class RegressionPrediction:
    """Output of one ``RegressionPredictor.predict()`` call."""

    expected_close: Optional[float]
    expected_high: Optional[float]
    expected_low: Optional[float]
    expected_return: Optional[float]
    expected_pip_move: Optional[float]
    expected_volatility: Optional[float]
    expected_MFE: Optional[float]
    expected_MAE: Optional[float]
    prediction_confidence: float
    prediction_interval: Dict[str, Tuple[float, float]]
    model_version: str
    feature_version: str
    training_version: str
    timestamp: str
    symbol: str
    timeframe: str
    prediction_horizon: int
    raw_predictions: Dict[str, float] = field(default_factory=dict)
    confidence_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    explanation: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # --- Decision Engine metadata (additive; averaged across configured
    # targets, same convention as prediction_confidence itself) -- see
    # model_scoring.py (training-time) and confidence.py (per-prediction). ---
    model_health_score: Optional[float] = None
    generalization_score: Optional[float] = None
    cross_validation_score: Optional[float] = None
    target_reliability: Optional[float] = None
    feature_quality_score: Optional[float] = None
    prediction_stability: Optional[float] = None
    distribution_shift_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_close": self.expected_close, "expected_high": self.expected_high,
            "expected_low": self.expected_low, "expected_return": self.expected_return,
            "expected_pip_move": self.expected_pip_move, "expected_volatility": self.expected_volatility,
            "expected_MFE": self.expected_MFE, "expected_MAE": self.expected_MAE,
            "prediction_confidence": round(self.prediction_confidence, 2),
            "prediction_interval": {
                k: (round(v[0], 6), round(v[1], 6)) for k, v in self.prediction_interval.items()
            },
            "model_version": self.model_version, "feature_version": self.feature_version,
            "training_version": self.training_version, "timestamp": self.timestamp,
            "symbol": self.symbol, "timeframe": self.timeframe,
            "prediction_horizon": self.prediction_horizon,
            "raw_predictions": self.raw_predictions,
            "confidence_breakdown": self.confidence_breakdown,
            "explanation": self.explanation, "warnings": self.warnings,
            "model_health_score": self.model_health_score, "generalization_score": self.generalization_score,
            "cross_validation_score": self.cross_validation_score, "target_reliability": self.target_reliability,
            "feature_quality_score": self.feature_quality_score, "prediction_stability": self.prediction_stability,
            "distribution_shift_score": self.distribution_shift_score,
        }


class RegressionPredictor:
    """Composes one or more loaded :class:`RegressionInferencePipeline`
    instances (one per target) into a single :class:`RegressionPrediction`."""

    def __init__(
        self, pipelines: Dict[str, RegressionInferencePipeline], feature_version: str = "1.0.0",
        pip_size: float = 0.0001,
    ) -> None:
        if not pipelines:
            raise ValueError("RegressionPredictor requires at least one loaded pipeline.")
        self.pipelines = pipelines
        self.feature_version = feature_version
        self.pip_size = pip_size

    def predict(self, market_state: MarketState, symbol: str = "UNKNOWN", timeframe: str = "UNKNOWN") -> RegressionPrediction:
        X, feature_names = extract_feature_vector(market_state)
        completeness = feature_completeness(market_state)

        raw_predictions: Dict[str, float] = {}
        intervals: Dict[str, Tuple[float, float]] = {}
        confidences: Dict[str, ConfidenceBreakdown] = {}
        health_scores: List[Any] = []
        warnings: List[str] = []

        for target, pipeline in self.pipelines.items():
            try:
                point, std = pipeline.predict(X, feature_names)
            except Exception as exc:  # noqa: BLE001 -- surface as a warning, keep other targets working
                warnings.append(f"{target}: prediction failed ({exc}).")
                continue

            value = float(np.ravel(point)[0])
            raw_predictions[target] = value
            model = pipeline.model
            assert model is not None

            z_scores = np.abs((X.ravel() - model.train_feature_mean_) / model.train_feature_std_)
            ensemble_std = float(np.ravel(std)[0]) if std is not None else None
            # Ratio computed here (the caller), never inside confidence.py --
            # that module never sees the predicted value itself, only this
            # one already-derived fraction (see confidence.py's module docstring).
            interval_width_fraction = (
                (2 * 1.96 * ensemble_std) / abs(value) if ensemble_std is not None and abs(value) > 1e-12 else None
            )

            breakdown = compute_confidence(
                residual_std=model.residual_std_, target_std=model.target_std_,
                test_r2=model.historical_r2_, feature_completeness_fraction=completeness,
                mean_abs_z_score=float(np.mean(z_scores)), ensemble_std=ensemble_std,
                cv_mean_r2=getattr(model, "cv_mean_r2_", None), cv_std_r2=getattr(model, "cv_std_r2_", None),
                interval_width_fraction=interval_width_fraction,
            )
            confidences[target] = breakdown

            model_health = getattr(model, "health_scores_", None)
            if model_health is not None:
                health_scores.append(model_health)

            if ensemble_std is not None:
                intervals[target] = (value - 1.96 * ensemble_std, value + 1.96 * ensemble_std)
            else:
                intervals[target] = (value, value)

        overall_confidence = float(np.mean([b.overall for b in confidences.values()])) if confidences else 0.0
        named = {field_name: raw_predictions.get(target) for target, field_name in TARGET_TO_PREDICTION_FIELD.items()}
        named_intervals = {TARGET_TO_PREDICTION_FIELD.get(t, t): v for t, v in intervals.items()}

        first_pipeline = next(iter(self.pipelines.values()))
        assert first_pipeline.model_version is not None
        horizon = first_pipeline.model_version.prediction_horizon
        model_versions = sorted({p.model_version.model_version for p in self.pipelines.values() if p.model_version})
        training_versions = sorted({
            p.model_version.version_info.training_pipeline_version
            for p in self.pipelines.values() if p.model_version
        })

        historical_rmse_pips = None
        pip_pipeline = next((p for t, p in self.pipelines.items() if t == "expected_pip_movement"), None)
        if pip_pipeline is not None and pip_pipeline.model is not None:
            historical_rmse_pips = pip_pipeline.model.historical_rmse_

        explanation = _generate_explanation(
            named=named, horizon=horizon, confidence=overall_confidence,
            intervals=named_intervals, historical_rmse_pips=historical_rmse_pips,
        )

        def _avg(attr: str) -> Optional[float]:
            values = [getattr(h, attr) for h in health_scores]
            return float(np.mean(values)) if values else None

        prediction_stability_avg = (
            float(np.mean([b.prediction_stability for b in confidences.values()])) if confidences else None
        )
        distribution_shift_avg = (
            float(np.mean([b.distribution_distance for b in confidences.values()])) if confidences else None
        )

        return RegressionPrediction(
            expected_close=named.get("expected_close"), expected_high=named.get("expected_high"),
            expected_low=named.get("expected_low"), expected_return=named.get("expected_return"),
            expected_pip_move=named.get("expected_pip_move"), expected_volatility=named.get("expected_volatility"),
            expected_MFE=named.get("expected_MFE"), expected_MAE=named.get("expected_MAE"),
            prediction_confidence=overall_confidence, prediction_interval=named_intervals,
            model_version=",".join(model_versions) or "unknown", feature_version=self.feature_version,
            training_version=",".join(training_versions) or "unknown", timestamp=utc_timestamp(),
            symbol=symbol, timeframe=timeframe, prediction_horizon=horizon,
            raw_predictions=raw_predictions,
            confidence_breakdown={t: b.to_dict() for t, b in confidences.items()},
            explanation=explanation, warnings=warnings,
            model_health_score=_avg("model_health_score"), generalization_score=_avg("generalization_score"),
            cross_validation_score=_avg("cross_validation_score"), target_reliability=_avg("target_reliability"),
            feature_quality_score=_avg("feature_quality_score"), prediction_stability=prediction_stability_avg,
            distribution_shift_score=distribution_shift_avg,
        )


def _generate_explanation(
    *, named: Dict[str, Optional[float]], horizon: int, confidence: float,
    intervals: Dict[str, Tuple[float, float]], historical_rmse_pips: Optional[float],
) -> List[str]:
    lines: List[str] = []
    pip_move = named.get("expected_pip_move")
    if pip_move is not None:
        lines.append(f"Expected movement: {pip_move:+.1f} pips")
    elif named.get("expected_return") is not None:
        lines.append(f"Expected return: {named['expected_return'] * 100:+.3f}%")
    lines.append(f"Prediction horizon: {horizon} candle{'s' if horizon != 1 else ''}")
    lines.append(f"Confidence: {confidence:.0f}%")
    if historical_rmse_pips is not None:
        lines.append(f"Model has historically achieved RMSE: {historical_rmse_pips:.1f} pips")
    if "expected_pip_move" in intervals:
        lower, upper = intervals["expected_pip_move"]
        half_width = (upper - lower) / 2.0
        lines.append(f"Prediction interval: +/-{half_width:.1f} pips")
    return lines
