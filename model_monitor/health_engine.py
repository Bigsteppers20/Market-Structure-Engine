"""Orchestrates every other monitoring module into one ``ModelHealthReport``
per model -- the concrete "MODEL HEALTH" deliverable.

Deriving each of ``health_score.py``'s 10 raw 0-100 factor scores from the
richer reports (``PerformanceReport``, ``CalibrationReport``,
``DriftReport``, ``ModelLifecycleMetadata``, ``PredictionLog.coverage()``)
is this module's entire job; ``health_score.compute_health_score()`` itself
stays a pure, independently-testable blending function.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .calibration_monitor import CalibrationReport
from .drift_detector import DriftReport
from .exceptions import InsufficientDataError
from .health_score import HealthScoreBreakdown, compute_health_score, severity_to_score
from .model_registry import ModelLifecycleMetadata
from .performance_monitor import PerformanceMonitor, RollingHistoricalPerformance
from .prediction_monitor import PredictionLog, ResolvedPrediction


@dataclass(slots=True)
class ModelHealthReport:
    """Complete health assessment for one monitored model."""

    model_name: str
    model_version: str
    task_type: str
    health: HealthScoreBreakdown
    status: str
    """``"good"``, ``"warning"``, or ``"critical"``."""
    performance: RollingHistoricalPerformance
    calibration: CalibrationReport
    drift: DriftReport
    lifecycle: ModelLifecycleMetadata
    coverage: float
    evaluated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name, "model_version": self.model_version, "task_type": self.task_type,
            "health": self.health.to_dict(), "status": self.status,
            "performance": self.performance.to_dict(), "calibration": self.calibration.to_dict(),
            "drift": self.drift.to_dict(), "lifecycle": self.lifecycle.to_dict(),
            "coverage": round(self.coverage, 4), "evaluated_at": self.evaluated_at,
        }


def _prediction_accuracy_score(performance) -> float:
    if performance.task_type == "regression":
        r2 = performance.metrics.get("r2", 0.0)
        r2 = 0.0 if r2 is None or np.isnan(r2) else r2
        return float(np.clip(r2, 0.0, 1.0) * 100.0)
    return float(performance.metrics.get("balanced_accuracy", performance.metrics.get("accuracy", 0.0)) * 100.0)


def _primary_error(performance) -> float:
    if performance.task_type == "regression":
        return float(performance.metrics.get("rmse", 0.0))
    return float(1.0 - performance.metrics.get("balanced_accuracy", performance.metrics.get("accuracy", 0.0)))


def _rolling_error_score(rh: RollingHistoricalPerformance) -> float:
    """100 if rolling (recent) error matches or beats historical error;
    degrades toward 0 as recent error grows to >= 2x the historical error."""
    historical_err = _primary_error(rh.historical)
    rolling_err = _primary_error(rh.rolling)
    if historical_err <= 1e-12:
        ratio = 1.0 if rolling_err <= 1e-12 else 2.0
    else:
        ratio = rolling_err / historical_err
    return float(np.clip(100.0 * (2.0 - ratio), 0.0, 100.0))


def _prediction_stability_score(
    resolved: Sequence[ResolvedPrediction], task_type: str, window: int, n_chunks: int = 4,
    regression_tolerance: Optional[float] = None,
) -> float:
    """Coefficient-of-variation of per-chunk error across ``n_chunks`` equal
    slices of the most recent ``window`` resolved predictions -- low
    variability across time = stable model. Neutral (70) when there isn't
    enough data yet to slice meaningfully."""
    recent = list(resolved)[-window:]
    if len(recent) < n_chunks * 2:
        return 70.0
    chunk_size = len(recent) // n_chunks
    chunk_errors: List[float] = []
    for i in range(n_chunks):
        chunk = recent[i * chunk_size: (i + 1) * chunk_size]
        if task_type == "regression":
            errs = [abs(r.error) for r in chunk if r.error is not None]
        else:
            errs = [0.0 if r.is_correct(regression_tolerance) else 1.0 for r in chunk if r.is_correct(regression_tolerance) is not None]
        if errs:
            chunk_errors.append(float(np.mean(errs)))
    if len(chunk_errors) < 2:
        return 70.0
    mean_err = float(np.mean(chunk_errors))
    cv = float(np.std(chunk_errors) / mean_err) if mean_err > 1e-12 else 0.0
    return float(np.clip(100.0 * (1.0 - min(cv, 1.0)), 0.0, 100.0))


class HealthEngine:
    """Computes one :class:`ModelHealthReport` per ``evaluate()`` call."""

    def __init__(self, config) -> None:
        self.config = config
        self._performance_monitor = PerformanceMonitor()

    def evaluate(
        self, *, model_name: str, model_version: str, task_type: str,
        prediction_log: PredictionLog, drift_report: DriftReport, calibration_report: CalibrationReport,
        lifecycle: ModelLifecycleMetadata, now_iso: str, classes: Optional[List[str]] = None,
        regression_tolerance: Optional[float] = None,
    ) -> ModelHealthReport:
        resolved = prediction_log.resolved
        if not resolved:
            raise InsufficientDataError("HealthEngine.evaluate() needs >= 1 resolved prediction.")

        rh = self._performance_monitor.rolling_vs_historical(
            resolved, task_type, self.config.rolling_window, classes=classes,
        )

        prediction_accuracy = _prediction_accuracy_score(rh.historical)
        rolling_error = _rolling_error_score(rh)
        prediction_stability = _prediction_stability_score(
            resolved, task_type, self.config.rolling_window, regression_tolerance=regression_tolerance,
        )
        confidence_calibration = 100.0 * (1.0 - min(calibration_report.calibration_error, 1.0))
        feature_drift_score = severity_to_score(drift_report.feature_drift.overall_severity)
        target_drift_score = severity_to_score(drift_report.target_drift.severity) if drift_report.target_drift else None
        residual_drift_score = severity_to_score(drift_report.residual_drift.severity) if drift_report.residual_drift else None
        market_regime_score = severity_to_score(drift_report.regime_drift.overall_shift) if drift_report.regime_drift else None
        training_age_score = float(np.clip(
            100.0 * (1.0 - lifecycle.model_age_days(now_iso) / self.config.max_model_age_days), 0.0, 100.0,
        ))
        prediction_coverage_score = prediction_log.coverage() * 100.0

        health = compute_health_score(
            prediction_accuracy=prediction_accuracy, prediction_stability=prediction_stability,
            confidence_calibration=confidence_calibration, feature_drift=feature_drift_score,
            rolling_error=rolling_error, prediction_coverage=prediction_coverage_score,
            target_drift=target_drift_score, residual_drift=residual_drift_score,
            market_regime_change=market_regime_score, training_age=training_age_score,
            weights=self.config.health_weights,
        )

        if health.overall >= self.config.health_threshold:
            status = "good"
        elif health.overall >= self.config.health_threshold * 0.5:
            status = "warning"
        else:
            status = "critical"

        return ModelHealthReport(
            model_name=model_name, model_version=model_version, task_type=task_type, health=health,
            status=status, performance=rh, calibration=calibration_report, drift=drift_report,
            lifecycle=lifecycle, coverage=prediction_log.coverage(), evaluated_at=now_iso,
        )
