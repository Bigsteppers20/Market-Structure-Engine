"""Retraining recommendation logic (RETRAINING RECOMMENDATION section).

Pure decision function: given a :class:`~model_monitor.health_engine.ModelHealthReport`,
a :class:`~model_monitor.drift_detector.DriftReport`, a
:class:`~model_monitor.model_registry.ModelLifecycleMetadata`, and the
count of newly-resolved samples since the last training run, decide
whether retraining is recommended -- and if so, with what priority,
reason(s), estimated benefit, and suggested dataset size/horizon. No side
effects, no I/O -- fully unit-testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .config import MonitorConfig
    from .drift_detector import DriftReport
    from .health_engine import ModelHealthReport
    from .model_registry import ModelLifecycleMetadata

PRIORITIES = ("none", "low", "medium", "high", "critical")


@dataclass(slots=True)
class RetrainingRecommendation:
    """Everything the spec requires a retraining recommendation to carry."""

    recommended: bool
    priority: str
    reasons: List[str] = field(default_factory=list)
    estimated_benefit: str = "unknown"
    suggested_dataset_size: int = 0
    suggested_horizon: int = 0

    def to_dict(self) -> dict:
        return {
            "recommended": self.recommended, "priority": self.priority, "reasons": self.reasons,
            "estimated_benefit": self.estimated_benefit, "suggested_dataset_size": self.suggested_dataset_size,
            "suggested_horizon": self.suggested_horizon,
        }


def _primary_error(performance) -> float:
    if performance.task_type == "regression":
        return float(performance.metrics.get("rmse", 0.0))
    return float(1.0 - performance.metrics.get("balanced_accuracy", performance.metrics.get("accuracy", 0.0)))


def _estimate_benefit(*, health_deficit: float, drift_severity: float) -> str:
    """A domain-informed heuristic, not a trained-model estimate (same
    disclosure convention as ``logistic_regression``'s Predictive Value
    tags): scales with how far health sits below 100 and how severe drift
    is. Purely illustrative for the Agentic AI's ``estimated_improvement``
    field -- never treated as a guarantee."""
    pct = max(0.0, min(40.0, health_deficit * 0.3 + drift_severity * 20.0))
    return f"+{pct:.0f}%"


def _check_health(health_report: "ModelHealthReport", config: "MonitorConfig"):
    triggered = health_report.health.overall < config.health_threshold
    reason = (
        f"Health score {health_report.health.overall:.1f} is below threshold {config.health_threshold:.1f}."
        if triggered else None
    )
    return triggered, reason


def _check_performance(health_report: "ModelHealthReport", config: "MonitorConfig"):
    rolling_err = _primary_error(health_report.performance.rolling)
    historical_err = _primary_error(health_report.performance.historical)
    if historical_err <= 1e-12:
        return False, None
    metric_key = "rmse_increase_pct" if health_report.task_type == "regression" else "accuracy_decrease_pct"
    threshold = config.performance_threshold.get(metric_key, 0.15)
    relative_increase = (rolling_err - historical_err) / historical_err
    if relative_increase <= threshold:
        return False, None
    metric_name = "RMSE" if health_report.task_type == "regression" else "error rate"
    reason = f"Rolling {metric_name} increased {relative_increase * 100:.1f}% vs. historical (threshold {threshold * 100:.0f}%)."
    return True, reason


def _check_feature_drift(drift_report: "DriftReport", config: "MonitorConfig"):
    fd = drift_report.feature_drift
    triggered = fd.overall_severity > config.feature_drift_threshold
    reason = (
        f"Feature drift severity {fd.overall_severity:.2f} exceeds threshold {config.feature_drift_threshold:.2f} "
        f"({fd.severity_label}); most drifted: {', '.join(fd.most_drifted[:5])}."
        if triggered else None
    )
    return triggered, reason


def _check_regime_shift(drift_report: "DriftReport"):
    regime = drift_report.regime_drift
    triggered = bool(regime and regime.shift_detected)
    reason = f"Market regime shift detected in: {', '.join(regime.differing_dimensions)}." if triggered else None
    return triggered, reason


def _check_model_age(lifecycle: "ModelLifecycleMetadata", config: "MonitorConfig", now_iso: str):
    age_days = lifecycle.model_age_days(now_iso)
    triggered = age_days > config.max_model_age_days
    reason = (
        f"Model age {age_days:.1f} days exceeds maximum {config.max_model_age_days:.1f} days." if triggered else None
    )
    return triggered, reason


def _priority_from_triggers(*, any_condition: bool, n_triggers: int, critical: bool) -> str:
    if not any_condition:
        return "none"
    if critical:
        return "critical"
    if n_triggers >= 3:
        return "high"
    if n_triggers == 2:
        return "medium"
    return "low"


def evaluate_retraining_policy(
    *, health_report: "ModelHealthReport", drift_report: "DriftReport",
    lifecycle: "ModelLifecycleMetadata", config: "MonitorConfig", new_sample_count: int, now_iso: str,
    current_horizon: int,
) -> RetrainingRecommendation:
    checks = [
        _check_health(health_report, config),
        _check_performance(health_report, config),
        _check_feature_drift(drift_report, config),
        _check_regime_shift(drift_report),
        _check_model_age(lifecycle, config, now_iso),
    ]
    triggers = [t for t, _ in checks]
    reasons = [r for _, r in checks if r is not None]

    any_condition = any(triggers)
    enough_data = new_sample_count >= config.min_new_samples
    recommended = any_condition and enough_data
    if any_condition and not enough_data:
        reasons.append(f"Insufficient new data ({new_sample_count} < {config.min_new_samples}) -- recommendation deferred.")

    critical = health_report.status == "critical" or drift_report.severity_label == "severe"
    priority = _priority_from_triggers(any_condition=any_condition, n_triggers=sum(triggers), critical=critical)

    health_deficit = max(0.0, config.health_threshold - health_report.health.overall)
    estimated_benefit = (
        _estimate_benefit(health_deficit=health_deficit, drift_severity=drift_report.overall_severity)
        if any_condition else "0%"
    )

    return RetrainingRecommendation(
        recommended=recommended, priority=priority, reasons=reasons, estimated_benefit=estimated_benefit,
        suggested_dataset_size=max(config.min_new_samples, lifecycle.training_dataset_size),
        suggested_horizon=current_horizon,
    )
