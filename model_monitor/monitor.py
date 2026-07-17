"""``ModelMonitor`` -- the production entry point tying every other module
together, mirroring the role ``LogisticRegressionEngine``/``RegressionEngine``
play for their own engines.

Live operation, entirely in memory::

    market_state = mse_engine.analyze()                     # Market Structure Engine
    prediction = engine.predict(market_state, ...)           # Linear or Logistic Regression Engine
    snapshot = ModelMonitor.from_regression_prediction(prediction, ...)
    monitor.record_prediction(snapshot)
    ...
    monitor.resolve_outcomes(df, resolver)                   # once horizons elapse
    health = monitor.evaluate_health()
    recommendation = monitor.recommend_retraining(current_horizon=5)
    report = monitor.to_agentic_report()                     # for the Agentic AI

This module never performs trade execution and never implements a Decision
Engine/Risk Manager -- its only output is health/drift/retraining reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from training.utils import utc_timestamp

from .calibration_monitor import CalibrationMonitor, CalibrationReport
from .config import MonitorConfig
from .drift_detector import DriftDetector, DriftReport, RegimeSnapshot, classify_regime
from .exceptions import InsufficientDataError, UnknownModelError
from .health_engine import HealthEngine, ModelHealthReport
from .model_registry import ModelLifecycleMetadata, ModelLifecycleRegistry
from .notification_manager import NotificationManager
from .prediction_monitor import (
    PredictionLog,
    PredictionSnapshot,
    ResolvedPrediction,
    Resolver,
    from_classification_prediction,
    from_regression_prediction,
)
from .retraining_manager import CandidateArtifact, RetrainingManager, RetrainingOutcome
from .retraining_policy import RetrainingRecommendation, evaluate_retraining_policy
from .validator import assert_valid_prediction_snapshot

TASK_TYPES = ("regression", "classification")


class ModelMonitor:
    """One monitor per production model (identified by ``model_name``,
    task-type-agnostic beyond the two-way branch every module already makes)."""

    def __init__(
        self, config: MonitorConfig, *, model_name: str, task_type: str,
        output_root: str | Path = "model_monitor_output", classes: Optional[Sequence[str]] = None,
    ) -> None:
        if task_type not in TASK_TYPES:
            raise ValueError(f"task_type={task_type!r}, expected one of {TASK_TYPES}.")
        self.config = config
        self.model_name = model_name
        self.task_type = task_type
        self.classes = list(classes) if classes else None

        self.lifecycle_registry = ModelLifecycleRegistry(output_root)
        self.notifications = NotificationManager(config.notification_policy)
        self.prediction_log = PredictionLog()
        self.drift_detector = DriftDetector(config)
        self.calibration_monitor = CalibrationMonitor()
        self.health_engine = HealthEngine(config)
        self.retraining_manager = RetrainingManager(config, self.lifecycle_registry, self.notifications)

        self._baseline_fitted = False
        self._last_health_report: Optional[ModelHealthReport] = None
        self._last_drift_report: Optional[DriftReport] = None
        self._last_calibration_report: Optional[CalibrationReport] = None
        self._last_recommendation: Optional[RetrainingRecommendation] = None

    # ------------------------------------------------------------------ #
    # Adapters -- convenience passthroughs to prediction_monitor's
    # module-level functions (kept there so they're usable without an
    # instantiated ModelMonitor too).
    # ------------------------------------------------------------------ #
    @staticmethod
    def from_regression_prediction(prediction: Any, **kwargs: Any) -> PredictionSnapshot:
        return from_regression_prediction(prediction, **kwargs)

    @staticmethod
    def from_classification_prediction(prediction: Any, **kwargs: Any) -> PredictionSnapshot:
        return from_classification_prediction(prediction, **kwargs)

    # ------------------------------------------------------------------ #
    # Lifecycle / baseline setup
    # ------------------------------------------------------------------ #
    def register_model(self, lifecycle: ModelLifecycleMetadata, *, as_production: bool = True) -> None:
        self.lifecycle_registry.register(lifecycle)
        if as_production:
            self.lifecycle_registry.promote(lifecycle.model_name, lifecycle.version, promotion_timestamp=lifecycle.training_date)

    def fit_baseline(
        self, X_train: np.ndarray, feature_names: Sequence[str], *,
        regime_snapshots: Optional[Sequence[RegimeSnapshot]] = None,
        target_values: Optional[np.ndarray] = None, residual_values: Optional[np.ndarray] = None,
    ) -> None:
        self.drift_detector.fit_baseline(
            X_train, feature_names, regime_snapshots=regime_snapshots,
            target_values=target_values, residual_values=residual_values,
        )
        self._baseline_fitted = True

    # ------------------------------------------------------------------ #
    # Prediction logging / outcome resolution
    # ------------------------------------------------------------------ #
    def record_prediction(self, snapshot: PredictionSnapshot, *, validate: bool = True) -> None:
        if validate:
            assert_valid_prediction_snapshot(snapshot)
        self.prediction_log.log(snapshot)
        try:
            self.lifecycle_registry.record_prediction(self.model_name, snapshot.model_version)
        except KeyError:
            pass  # model not yet registered -- still tracked locally in prediction_log

    def resolve_outcomes(self, df, resolver: Resolver, now_iso: Optional[str] = None) -> List[ResolvedPrediction]:
        return self.prediction_log.resolve(df, resolver, now_iso or utc_timestamp())

    # ------------------------------------------------------------------ #
    def _lifecycle_for(self, version: Optional[str]) -> ModelLifecycleMetadata:
        if version is not None:
            return self.lifecycle_registry.get(self.model_name, version)
        lifecycle = self.lifecycle_registry.production_version(self.model_name)
        if lifecycle is None:
            raise UnknownModelError(f"No production model registered for {self.model_name!r}.")
        return lifecycle

    def _current_regimes(self, window: int) -> List[RegimeSnapshot]:
        snapshots = self.prediction_log.recent_snapshots(window)
        return [classify_regime(dict(zip(s.feature_names, s.feature_vector))) for s in snapshots]

    def _current_target_and_residual_series(self, window: int):
        rolling = self.prediction_log.rolling(window)
        if self.task_type == "regression":
            targets = [r.actual_value for r in rolling if r.actual_value is not None]
            residuals = [r.error for r in rolling if r.error is not None]
        else:
            class_to_idx = {c: i for i, c in enumerate(self.classes)} if self.classes else {}
            targets = [class_to_idx[r.actual_class] for r in rolling if r.actual_class in class_to_idx]
            residuals = [r.classification_residual for r in rolling if r.classification_residual is not None]
        targets_arr = np.asarray(targets, dtype=float) if len(targets) >= 2 else None
        residuals_arr = np.asarray(residuals, dtype=float) if len(residuals) >= 2 else None
        return targets_arr, residuals_arr

    # ------------------------------------------------------------------ #
    # Health / drift
    # ------------------------------------------------------------------ #
    def evaluate_health(
        self, *, now_iso: Optional[str] = None, model_version: Optional[str] = None,
        regression_tolerance: Optional[float] = None,
    ) -> ModelHealthReport:
        if not self._baseline_fitted:
            raise InsufficientDataError("Call fit_baseline() before evaluate_health().")
        now_iso = now_iso or utc_timestamp()
        lifecycle = self._lifecycle_for(model_version)
        resolved = self.prediction_log.resolved
        if not resolved:
            raise InsufficientDataError("evaluate_health() needs >= 1 resolved prediction -- call resolve_outcomes() first.")

        calibration_report = self.calibration_monitor.evaluate(resolved, regression_tolerance=regression_tolerance)

        recent_snapshots = self.prediction_log.recent_snapshots(self.config.market_regime_lookback)
        X_live = np.array([s.feature_vector for s in recent_snapshots], dtype=float)
        feature_names = recent_snapshots[-1].feature_names
        valid_mask = None
        if all(s.valid_mask is not None for s in recent_snapshots):
            valid_mask = np.array([s.valid_mask for s in recent_snapshots], dtype=bool)
        current_regimes = self._current_regimes(self.config.market_regime_lookback)
        current_targets, current_residuals = self._current_target_and_residual_series(self.config.rolling_window)

        drift_report = self.drift_detector.detect(
            X_live, feature_names, valid_mask=valid_mask, current_regimes=current_regimes,
            current_targets=current_targets, current_residuals=current_residuals,
        )

        health_report = self.health_engine.evaluate(
            model_name=self.model_name, model_version=lifecycle.version, task_type=self.task_type,
            prediction_log=self.prediction_log, drift_report=drift_report, calibration_report=calibration_report,
            lifecycle=lifecycle, now_iso=now_iso, classes=self.classes, regression_tolerance=regression_tolerance,
        )

        if health_report.status in ("warning", "critical"):
            self.notifications.notify_model_health_warning(
                model_name=self.model_name, health_score=health_report.health.overall, status=health_report.status,
            )
        if drift_report.feature_drift.severity_label in ("high", "severe"):
            self.notifications.notify_feature_drift_alert(
                model_name=self.model_name, severity_label=drift_report.feature_drift.severity_label,
                overall_severity=drift_report.feature_drift.overall_severity,
            )

        self._last_health_report = health_report
        self._last_drift_report = drift_report
        self._last_calibration_report = calibration_report
        return health_report

    # ------------------------------------------------------------------ #
    # Retraining
    # ------------------------------------------------------------------ #
    def recommend_retraining(
        self, *, current_horizon: int, now_iso: Optional[str] = None, new_sample_count: Optional[int] = None,
    ) -> RetrainingRecommendation:
        now_iso = now_iso or utc_timestamp()
        if self._last_health_report is None or self._last_drift_report is None:
            self.evaluate_health(now_iso=now_iso)
        assert self._last_health_report is not None and self._last_drift_report is not None
        lifecycle = self._lifecycle_for(self._last_health_report.model_version)
        new_sample_count = new_sample_count if new_sample_count is not None else len(self.prediction_log.resolved)
        recommendation = evaluate_retraining_policy(
            health_report=self._last_health_report, drift_report=self._last_drift_report, lifecycle=lifecycle,
            config=self.config, new_sample_count=new_sample_count, now_iso=now_iso, current_horizon=current_horizon,
        )
        self._last_recommendation = recommendation
        return recommendation

    def check_retraining(
        self, *, current_horizon: int, now_iso: Optional[str] = None, new_sample_count: Optional[int] = None,
        train_candidate_fn: Optional[Callable[[], CandidateArtifact]] = None,
    ) -> RetrainingOutcome:
        now_iso = now_iso or utc_timestamp()
        recommendation = self.recommend_retraining(current_horizon=current_horizon, now_iso=now_iso, new_sample_count=new_sample_count)
        assert self._last_health_report is not None and self._last_drift_report is not None
        lifecycle = self._lifecycle_for(self._last_health_report.model_version)
        new_sample_count = new_sample_count if new_sample_count is not None else len(self.prediction_log.resolved)
        return self.retraining_manager.run_cycle(
            model_name=self.model_name, task_type=self.task_type, mode=self.config.retraining_mode,
            health_report=self._last_health_report, drift_report=self._last_drift_report,
            recommendation=recommendation, last_trained_iso=lifecycle.training_date, now_iso=now_iso,
            new_sample_count=new_sample_count, train_candidate_fn=train_candidate_fn,
        )

    # ------------------------------------------------------------------ #
    # Agentic AI integration
    # ------------------------------------------------------------------ #
    def to_agentic_report(self, *, model_display_name: Optional[str] = None) -> Dict[str, Any]:
        """Exact shape of the spec's AGENTIC AI INTEGRATION example."""
        if self._last_health_report is None:
            raise InsufficientDataError("Call evaluate_health() before to_agentic_report().")
        health_report = self._last_health_report
        recommendation = self._last_recommendation

        calibration_status = (
            "Good" if health_report.calibration.status == "well_calibrated"
            else health_report.calibration.status.replace("_", " ").title()
        )
        return {
            "model": model_display_name or self.model_name,
            "health_score": round(health_report.health.overall),
            "status": health_report.status.title(),
            "feature_drift": round(health_report.drift.feature_drift.overall_severity, 2),
            "calibration": calibration_status,
            "retraining_recommended": bool(recommendation.recommended) if recommendation else False,
            "estimated_improvement": recommendation.estimated_benefit if recommendation else "0%",
            "priority": recommendation.priority.title() if recommendation else "None",
        }
