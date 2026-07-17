"""Prediction logging and outcome resolution.

This is the model-agnostic core the rest of ``model_monitor`` operates on:
:class:`PredictionSnapshot` normalizes whatever a concrete engine's own
prediction type looks like (``linear_regression.RegressionPrediction``,
``logistic_regression.ClassificationPrediction``, or a future model's own
type) into one common shape. A future model plugs into this system by
either constructing a :class:`PredictionSnapshot` directly or supplying its
own adapter function analogous to :func:`from_regression_prediction`/
:func:`from_classification_prediction` -- nothing downstream
(``performance_monitor``, ``calibration_monitor``, ``drift_detector``,
``health_score``) ever imports or inspects a concrete engine's prediction
type.

Outcome resolution deliberately reuses the resolver-callable shape
``ml_pipeline.label_generator`` already established: a real
``LabelGenerator`` instance's bound ``.label(df, index, horizon)`` method
(classification) can be passed to :meth:`PredictionLog.resolve` directly,
with zero adaptation. A regression-target function from
``ml_pipeline.label_generator.REGRESSION_REGISTRY`` takes a 4th
``pip_size`` argument, so bind it first --
``functools.partial(REGRESSION_REGISTRY["next_close"], pip_size=0.0001)``
-- ground-truth computation is still never duplicated, just bound to a
3-arg callable at the call site.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

Resolver = Callable[[pd.DataFrame, int, int], Any]


@dataclass(slots=True)
class PredictionSnapshot:
    """One live prediction, normalized to a model-agnostic shape."""

    task_type: str
    """``"regression"`` or ``"classification"`` -- the only field any
    downstream module is allowed to branch on."""
    model_name: str
    model_version: str
    feature_version: str
    training_version: str
    symbol: str
    timeframe: str
    prediction_horizon: int
    timestamp: str
    decision_index: int
    """Row index into the raw OHLCV DataFrame this prediction was made at --
    the anchor outcome resolution walks forward from."""
    feature_vector: List[float]
    feature_names: List[str]
    confidence: float
    """0-100, both existing engines' ``prediction_confidence``."""
    predicted_value: Optional[float] = None
    raw_predictions: Dict[str, float] = field(default_factory=dict)
    primary_target: Optional[str] = None
    predicted_class: Optional[str] = None
    class_probabilities: Optional[Dict[str, float]] = None
    valid_mask: Optional[List[bool]] = None
    """Per-feature ``_valid`` companion flags, if the caller supplied them --
    feeds ``feature_drift``'s missing-feature-drift diagnostic."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type, "model_name": self.model_name, "model_version": self.model_version,
            "feature_version": self.feature_version, "training_version": self.training_version,
            "symbol": self.symbol, "timeframe": self.timeframe, "prediction_horizon": self.prediction_horizon,
            "timestamp": self.timestamp, "decision_index": self.decision_index, "confidence": self.confidence,
            "predicted_value": self.predicted_value, "raw_predictions": self.raw_predictions,
            "primary_target": self.primary_target, "predicted_class": self.predicted_class,
            "class_probabilities": self.class_probabilities,
        }


@dataclass(slots=True)
class ResolvedPrediction:
    """A :class:`PredictionSnapshot` paired with its now-known actual outcome."""

    snapshot: PredictionSnapshot
    resolved_at: str
    actual_value: Optional[float] = None
    actual_class: Optional[str] = None

    @property
    def error(self) -> Optional[float]:
        """``actual - predicted`` (regression only)."""
        if self.actual_value is None or self.snapshot.predicted_value is None:
            return None
        return self.actual_value - self.snapshot.predicted_value

    @property
    def classification_residual(self) -> Optional[float]:
        """``1 - P(actual class)`` -- a classification analogue of a
        residual magnitude, feeding ``drift_detector``'s residual-drift
        diagnostic without that module needing to know about probabilities."""
        if self.actual_class is None or not self.snapshot.class_probabilities:
            return None
        return 1.0 - self.snapshot.class_probabilities.get(self.actual_class, 0.0)

    def is_correct(self, regression_tolerance: Optional[float] = None) -> Optional[bool]:
        """Classification: exact predicted/actual class match. Regression
        (no ``regression_tolerance``): directional correctness (same sign)
        -- meaningful for trading-style continuous targets. Regression
        (with ``regression_tolerance``): ``|error| <= tolerance``."""
        if self.snapshot.task_type == "classification":
            if self.actual_class is None:
                return None
            return self.snapshot.predicted_class == self.actual_class
        if self.error is None:
            return None
        if regression_tolerance is not None:
            return abs(self.error) <= regression_tolerance
        return (self.snapshot.predicted_value >= 0) == (self.actual_value >= 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(), "resolved_at": self.resolved_at,
            "actual_value": self.actual_value, "actual_class": self.actual_class,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Adapters -- the ONLY place this system touches a concrete engine's
# prediction type. Future models supply an analogous function.
# --------------------------------------------------------------------------- #
def from_regression_prediction(
    prediction: Any, *, model_name: str, primary_target: str, decision_index: int,
    feature_vector: Sequence[float], feature_names: Sequence[str], valid_mask: Optional[Sequence[bool]] = None,
) -> PredictionSnapshot:
    """Adapt a ``linear_regression.RegressionPrediction`` (duck-typed --
    never imports ``linear_regression`` itself, keeping this system usable
    even if that engine isn't installed)."""
    return PredictionSnapshot(
        task_type="regression", model_name=model_name, model_version=prediction.model_version,
        feature_version=prediction.feature_version, training_version=prediction.training_version,
        symbol=prediction.symbol, timeframe=prediction.timeframe,
        prediction_horizon=prediction.prediction_horizon, timestamp=prediction.timestamp,
        decision_index=decision_index, feature_vector=list(feature_vector), feature_names=list(feature_names),
        confidence=prediction.prediction_confidence, predicted_value=prediction.raw_predictions.get(primary_target),
        raw_predictions=dict(prediction.raw_predictions), primary_target=primary_target,
        valid_mask=list(valid_mask) if valid_mask is not None else None,
    )


def from_classification_prediction(
    prediction: Any, *, model_name: str, decision_index: int,
    feature_vector: Sequence[float], feature_names: Sequence[str], valid_mask: Optional[Sequence[bool]] = None,
) -> PredictionSnapshot:
    """Adapt a ``logistic_regression.ClassificationPrediction`` (duck-typed,
    same rationale as :func:`from_regression_prediction`)."""
    return PredictionSnapshot(
        task_type="classification", model_name=model_name, model_version=prediction.model_version,
        feature_version=prediction.feature_version, training_version=prediction.training_version,
        symbol=prediction.symbol, timeframe=prediction.timeframe,
        prediction_horizon=prediction.prediction_horizon, timestamp=prediction.timestamp,
        decision_index=decision_index, feature_vector=list(feature_vector), feature_names=list(feature_names),
        confidence=prediction.prediction_confidence, predicted_class=prediction.predicted_class,
        class_probabilities=dict(prediction.class_probabilities),
        valid_mask=list(valid_mask) if valid_mask is not None else None,
    )


# --------------------------------------------------------------------------- #
# PredictionLog
# --------------------------------------------------------------------------- #
class PredictionLog:
    """In-memory log of predictions, pending until their outcome can be
    resolved from later data."""

    def __init__(self) -> None:
        self._pending: List[PredictionSnapshot] = []
        self._resolved: List[ResolvedPrediction] = []
        self._all_snapshots: List[PredictionSnapshot] = []

    def log(self, snapshot: PredictionSnapshot) -> None:
        self._pending.append(snapshot)
        self._all_snapshots.append(snapshot)

    def resolve(self, df: pd.DataFrame, resolver: Resolver, now_iso: str) -> List[ResolvedPrediction]:
        """Resolve every pending prediction whose horizon has fully
        elapsed within ``df`` -- i.e. ``decision_index + horizon < len(df)``.
        ``resolver`` is called as ``resolver(df, decision_index, horizon)``,
        exactly the signature of an ``ml_pipeline.label_generator``
        regression-target function or a ``LabelGenerator.label`` bound
        method -- pass one of those directly."""
        newly_resolved: List[ResolvedPrediction] = []
        still_pending: List[PredictionSnapshot] = []
        for snapshot in self._pending:
            end_index = snapshot.decision_index + snapshot.prediction_horizon
            if end_index >= len(df):
                still_pending.append(snapshot)
                continue
            outcome = resolver(df, snapshot.decision_index, snapshot.prediction_horizon)
            if snapshot.task_type == "classification":
                resolved = ResolvedPrediction(snapshot=snapshot, resolved_at=now_iso, actual_class=str(outcome))
            else:
                resolved = ResolvedPrediction(snapshot=snapshot, resolved_at=now_iso, actual_value=float(outcome))
            newly_resolved.append(resolved)
            self._resolved.append(resolved)
        self._pending = still_pending
        return newly_resolved

    def coverage(self) -> float:
        """Fraction of logged predictions that have a resolved outcome --
        the PREDICTION COVERAGE health factor."""
        total = len(self._pending) + len(self._resolved)
        return len(self._resolved) / total if total else 1.0

    def rolling(self, window: int) -> List[ResolvedPrediction]:
        return self._resolved[-window:] if window > 0 else []

    def recent_snapshots(self, window: int) -> List[PredictionSnapshot]:
        """Most recently *logged* snapshots (pending or resolved, in log
        order) -- feeds feature/regime drift, which only needs the input
        feature vector, not a resolved outcome."""
        return self._all_snapshots[-window:] if window > 0 else []

    @property
    def resolved(self) -> List[ResolvedPrediction]:
        return list(self._resolved)

    @property
    def pending(self) -> List[PredictionSnapshot]:
        return list(self._pending)

    def __len__(self) -> int:
        return len(self._pending) + len(self._resolved)
