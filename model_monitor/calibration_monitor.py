"""Confidence-vs-accuracy calibration monitoring (CALIBRATION MONITOR section).

Operates purely on :class:`~model_monitor.prediction_monitor.ResolvedPrediction`
objects, so it works identically for a regression model (correctness =
``ResolvedPrediction.is_correct(regression_tolerance)``, directional or
tolerance-based) and a classification model (exact class match) -- never
branches on task type itself. This is a coarser-grained sibling of
``logistic_regression.calibration``'s probability-level ECE: that module
calibrates a *single class's* probability against observed frequency at
training/evaluation time; this one tracks the engine's overall 0-100
``prediction_confidence`` against observed correctness continuously in
production, across both task types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .exceptions import InsufficientDataError
from .prediction_monitor import ResolvedPrediction


@dataclass(slots=True)
class ConfidenceBucket:
    """One confidence-decile's stated-vs-observed comparison."""

    bucket_range: Tuple[float, float]
    n_samples: int
    mean_confidence: float
    observed_accuracy: float
    gap: float
    """``mean_confidence - observed_accuracy`` -- positive = overconfident,
    negative = underconfident, in this bucket."""

    def to_dict(self) -> dict:
        return {
            "bucket_range": list(self.bucket_range), "n_samples": self.n_samples,
            "mean_confidence": round(self.mean_confidence, 2), "observed_accuracy": round(self.observed_accuracy, 2),
            "gap": round(self.gap, 2),
        }


@dataclass(slots=True)
class CalibrationReport:
    """Full confidence-calibration assessment for one batch of resolved predictions."""

    buckets: List[ConfidenceBucket] = field(default_factory=list)
    overall_mean_confidence: float = 0.0
    overall_observed_accuracy: float = 0.0
    calibration_error: float = 0.0
    """Sample-weighted mean |gap| / 100 -- in [0, 1], analogous to ECE."""
    status: str = "well_calibrated"
    """``"overconfident"``, ``"underconfident"``, or ``"well_calibrated"``."""
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "buckets": [b.to_dict() for b in self.buckets],
            "overall_mean_confidence": round(self.overall_mean_confidence, 2),
            "overall_observed_accuracy": round(self.overall_observed_accuracy, 2),
            "calibration_error": round(self.calibration_error, 4),
            "status": self.status, "n_samples": self.n_samples,
        }


class CalibrationMonitor:
    """Buckets resolved predictions by stated confidence and compares
    against observed correctness."""

    def __init__(self, n_buckets: int = 10, overconfidence_threshold: float = 10.0) -> None:
        self.n_buckets = n_buckets
        self.overconfidence_threshold = overconfidence_threshold

    def evaluate(
        self, resolved: Sequence[ResolvedPrediction], regression_tolerance: Optional[float] = None,
    ) -> CalibrationReport:
        confidences: List[float] = []
        correctness: List[bool] = []
        for r in resolved:
            is_correct = r.is_correct(regression_tolerance)
            if is_correct is None:
                continue
            confidences.append(r.snapshot.confidence)
            correctness.append(bool(is_correct))

        if not confidences:
            raise InsufficientDataError("CalibrationMonitor.evaluate() needs >= 1 resolved prediction with a known outcome.")

        conf_arr = np.asarray(confidences, dtype=float)
        correct_arr = np.asarray(correctness, dtype=float)
        edges = np.linspace(0.0, 100.0, self.n_buckets + 1)
        bucket_ids = np.clip(np.digitize(conf_arr, edges[1:-1]), 0, self.n_buckets - 1)

        buckets: List[ConfidenceBucket] = []
        for b in range(self.n_buckets):
            mask = bucket_ids == b
            if not mask.any():
                continue
            mean_confidence = float(conf_arr[mask].mean())
            observed_accuracy = float(correct_arr[mask].mean()) * 100.0
            buckets.append(ConfidenceBucket(
                bucket_range=(float(edges[b]), float(edges[b + 1])), n_samples=int(mask.sum()),
                mean_confidence=mean_confidence, observed_accuracy=observed_accuracy,
                gap=mean_confidence - observed_accuracy,
            ))

        overall_mean_confidence = float(conf_arr.mean())
        overall_observed_accuracy = float(correct_arr.mean()) * 100.0
        total = sum(b.n_samples for b in buckets)
        calibration_error = (
            sum(b.n_samples * abs(b.gap) for b in buckets) / total / 100.0 if total else 0.0
        )

        overall_gap = overall_mean_confidence - overall_observed_accuracy
        if overall_gap > self.overconfidence_threshold:
            status = "overconfident"
        elif overall_gap < -self.overconfidence_threshold:
            status = "underconfident"
        else:
            status = "well_calibrated"

        return CalibrationReport(
            buckets=buckets, overall_mean_confidence=overall_mean_confidence,
            overall_observed_accuracy=overall_observed_accuracy, calibration_error=calibration_error,
            status=status, n_samples=len(confidences),
        )


def detect_calibration_drift(baseline: CalibrationReport, current: CalibrationReport) -> float:
    """Absolute change in calibration error between two reports (e.g. a
    training-time/early-production baseline vs. the current rolling
    window) -- the CALIBRATION DRIFT diagnostic."""
    return abs(current.calibration_error - baseline.calibration_error)
