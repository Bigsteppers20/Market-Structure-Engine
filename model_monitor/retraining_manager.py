"""Retraining orchestration and candidate promotion (RETRAINING SAFETY /
MODEL COMPARISON sections).

Automatic retraining never overwrites a production model in place: a
candidate is trained (via a caller-supplied callback -- this module never
imports ``linear_regression``/``logistic_regression`` training code
directly, keeping it usable by any future model family), registered under
its own version with ``status="candidate"``, evaluated, and compared
against the current production model's real-world (rolling+historical)
performance. Only a candidate that clears the configured
:class:`~model_monitor.config.PromotionPolicy` is promoted --
``ModelLifecycleRegistry.promote()`` archives (never deletes) whatever was
previously production, so the full version history always remains
intact and retrievable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import MonitorConfig, PromotionPolicy
from .drift_detector import DriftReport
from .exceptions import PromotionError, RetrainingError
from .health_engine import ModelHealthReport
from .model_registry import ModelLifecycleMetadata, ModelLifecycleRegistry
from .notification_manager import NotificationManager
from .retraining_policy import RetrainingRecommendation
from .retraining_scheduler import should_trigger_retraining

#: Direction each comparable metric should move to count as an
#: "improvement" -- the MODEL COMPARISON section's Accuracy/RMSE/MAE/F1/
#: Calibration list, plus ROC-AUC/balanced_accuracy as classification
#: fallbacks. Extend this mapping (not the comparison logic) for a new metric.
METRIC_DIRECTIONS: Dict[str, str] = {
    "rmse": "lower", "mae": "lower", "mape": "lower", "log_loss": "lower", "brier_score": "lower",
    "calibration_error": "lower",
    "r2": "higher", "explained_variance": "higher", "accuracy": "higher", "balanced_accuracy": "higher",
    "f1": "higher", "roc_auc": "higher", "pr_auc": "higher",
}


@dataclass(slots=True)
class CandidateArtifact:
    """Everything the promotion workflow needs about a freshly-trained
    candidate model. Constructed by the caller's ``train_candidate_fn``
    (this module never trains anything itself)."""

    version: str
    metrics: Dict[str, float]
    training_date: str
    training_dataset_size: int
    feature_version: str = "1.0.0"
    training_version: str = "1.0.0"
    strategy_version: str = "1.0.0"
    dataset_version: str = "1.0.0"
    supported_symbols: List[str] = field(default_factory=list)
    supported_timeframes: List[str] = field(default_factory=list)
    calibration_error: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    artifact_dir: str = ""


@dataclass(slots=True)
class ComparisonResult:
    """MODEL COMPARISON output: Deploy / Reject / Further Evaluation."""

    decision: str
    """``"deploy"``, ``"reject"``, or ``"further_evaluation"``."""
    primary_metric: str
    production_value: float
    candidate_value: float
    relative_improvement: float
    reasons: List[str] = field(default_factory=list)
    feature_importance_changes: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision, "primary_metric": self.primary_metric,
            "production_value": self.production_value, "candidate_value": self.candidate_value,
            "relative_improvement": round(self.relative_improvement, 4), "reasons": self.reasons,
            "feature_importance_changes": self.feature_importance_changes,
        }


def _relative_improvement(*, production_value: float, candidate_value: float, direction: str) -> float:
    if direction == "lower":
        return (production_value - candidate_value) / abs(production_value) if production_value else (
            1.0 if candidate_value < production_value else 0.0
        )
    return (candidate_value - production_value) / abs(production_value) if production_value else (
        1.0 if candidate_value > production_value else 0.0
    )


def compare_models(
    *, task_type: str, production: CandidateArtifact, candidate: CandidateArtifact, policy: PromotionPolicy,
) -> ComparisonResult:
    """Compare a candidate against the current production model and
    recommend Deploy / Reject / Further Evaluation."""
    default_primary = "rmse" if task_type == "regression" else "balanced_accuracy"
    common_metrics = [m for m in METRIC_DIRECTIONS if m in production.metrics and m in candidate.metrics]
    if not common_metrics:
        raise PromotionError(
            f"No comparable metric found between production ({sorted(production.metrics)}) and "
            f"candidate ({sorted(candidate.metrics)})."
        )
    primary = default_primary if default_primary in common_metrics else common_metrics[0]
    direction = METRIC_DIRECTIONS[primary]
    production_value, candidate_value = production.metrics[primary], candidate.metrics[primary]
    relative_improvement = _relative_improvement(
        production_value=production_value, candidate_value=candidate_value, direction=direction,
    )

    reasons: List[str] = []
    if production.inference_latency_ms and candidate.inference_latency_ms:
        latency_regression = (candidate.inference_latency_ms - production.inference_latency_ms) / production.inference_latency_ms
        if latency_regression > policy.max_inference_latency_regression:
            reasons.append(
                f"Inference latency regressed {latency_regression * 100:.1f}% "
                f"(max allowed {policy.max_inference_latency_regression * 100:.0f}%)."
            )

    feature_importance_changes = None
    if production.feature_importance and candidate.feature_importance:
        names = set(production.feature_importance) | set(candidate.feature_importance)
        feature_importance_changes = {
            name: candidate.feature_importance.get(name, 0.0) - production.feature_importance.get(name, 0.0)
            for name in names
        }

    if reasons:
        decision = "reject"
    elif relative_improvement >= policy.min_relative_improvement:
        decision = "deploy"
        reasons.append(
            f"{primary} improved {relative_improvement * 100:.2f}% (>= required {policy.min_relative_improvement * 100:.2f}%)."
        )
    elif abs(relative_improvement) <= policy.tie_tolerance:
        if policy.allow_tie_promotion:
            decision = "deploy"
            reasons.append(f"{primary} tied (within {policy.tie_tolerance * 100:.2f}%) -- tie promotion allowed by policy.")
        else:
            decision = "further_evaluation"
            reasons.append(f"{primary} tied (within {policy.tie_tolerance * 100:.2f}%) -- policy requires further evaluation on ties.")
    else:
        decision = "reject"
        reasons.append(
            f"{primary} did not improve enough ({relative_improvement * 100:.2f}% < required {policy.min_relative_improvement * 100:.2f}%)."
        )

    return ComparisonResult(
        decision=decision, primary_metric=primary, production_value=production_value,
        candidate_value=candidate_value, relative_improvement=relative_improvement, reasons=reasons,
        feature_importance_changes=feature_importance_changes,
    )


@dataclass(slots=True)
class RetrainingOutcome:
    """What happened during one ``RetrainingManager.run_cycle()`` call."""

    triggered: bool
    reason: str
    comparison: Optional[ComparisonResult] = None
    promoted_version: Optional[str] = None
    candidate_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered, "reason": self.reason,
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "promoted_version": self.promoted_version, "candidate_version": self.candidate_version,
        }


class RetrainingManager:
    """Ties scheduling, candidate training, comparison, promotion, and
    notification together into one workflow."""

    def __init__(
        self, config: MonitorConfig, lifecycle_registry: ModelLifecycleRegistry,
        notification_manager: Optional[NotificationManager] = None,
    ) -> None:
        self.config = config
        self.lifecycle_registry = lifecycle_registry
        self.notifications = notification_manager or NotificationManager(config.notification_policy)

    def run_cycle(
        self, *, model_name: str, task_type: str, mode: str, health_report: ModelHealthReport,
        drift_report: DriftReport, recommendation: RetrainingRecommendation, last_trained_iso: str,
        now_iso: str, new_sample_count: int,
        train_candidate_fn: Optional[Callable[[], CandidateArtifact]] = None,
    ) -> RetrainingOutcome:
        if recommendation.recommended:
            self.notifications.notify_retraining_recommended(
                model_name=model_name, priority=recommendation.priority, reasons=recommendation.reasons,
            )

        triggered, reason = should_trigger_retraining(
            mode=mode, schedule=self.config.retraining_schedule, health_report=health_report,
            drift_report=drift_report, config=self.config, last_trained_iso=last_trained_iso,
            now_iso=now_iso, new_sample_count=new_sample_count,
        )
        if not triggered:
            return RetrainingOutcome(triggered=False, reason=reason)

        if train_candidate_fn is None:
            raise RetrainingError(
                f"Retraining was triggered ({reason}) but no train_candidate_fn was supplied to run_cycle()."
            )

        self.notifications.notify_retraining_started(model_name=model_name)
        candidate = train_candidate_fn()
        self.lifecycle_registry.register(ModelLifecycleMetadata(
            model_name=model_name, version=candidate.version, task_type=task_type, status="candidate",
            training_date=candidate.training_date, training_dataset_size=candidate.training_dataset_size,
            feature_version=candidate.feature_version, training_version=candidate.training_version,
            strategy_version=candidate.strategy_version, dataset_version=candidate.dataset_version,
            supported_symbols=candidate.supported_symbols, supported_timeframes=candidate.supported_timeframes,
            artifact_dir=candidate.artifact_dir,
        ))
        self.notifications.notify_retraining_completed(model_name=model_name, candidate_version=candidate.version)

        production_meta = self.lifecycle_registry.production_version(model_name)
        if production_meta is None:
            self.lifecycle_registry.promote(model_name, candidate.version, promotion_timestamp=now_iso)
            self.notifications.notify_candidate_promoted(
                model_name=model_name, version=candidate.version, reasons=["No existing production model -- promoted unconditionally."],
            )
            return RetrainingOutcome(
                triggered=True, reason=reason, promoted_version=candidate.version, candidate_version=candidate.version,
            )

        production_artifact = CandidateArtifact(
            version=production_meta.version, metrics=health_report.performance.historical.metrics,
            training_date=production_meta.training_date, training_dataset_size=production_meta.training_dataset_size,
            calibration_error=health_report.calibration.calibration_error,
        )
        comparison = compare_models(task_type=task_type, production=production_artifact, candidate=candidate, policy=self.config.promotion_policy)

        if comparison.decision == "deploy":
            self.lifecycle_registry.promote(model_name, candidate.version, promotion_timestamp=now_iso)
            self.notifications.notify_candidate_promoted(model_name=model_name, version=candidate.version, reasons=comparison.reasons)
            self.notifications.notify_performance_improved(model_name=model_name, relative_improvement=comparison.relative_improvement)
            promoted_version = candidate.version
        else:
            self.lifecycle_registry.archive(model_name, candidate.version)
            self.notifications.notify_candidate_rejected(model_name=model_name, version=candidate.version, reasons=comparison.reasons)
            if comparison.relative_improvement < 0:
                self.notifications.notify_performance_degraded(model_name=model_name, relative_change=comparison.relative_improvement)
            promoted_version = None

        return RetrainingOutcome(
            triggered=True, reason=reason, comparison=comparison,
            promoted_version=promoted_version, candidate_version=candidate.version,
        )
