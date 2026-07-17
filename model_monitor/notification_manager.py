"""Structured notification generation (NOTIFICATION SYSTEM section).

No real delivery channel (email/Slack/etc.) is implemented here -- the
spec's requirement is "generate structured notifications", and the
Agentic AI integration point is ``monitor.py``'s ``to_agentic_report()``.
:meth:`NotificationManager.add_handler` lets a caller wire in whatever
delivery mechanism it wants (log line, webhook, message queue, ...)
without this module knowing anything about it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from training.utils import utc_timestamp

from .config import NotificationPolicy

NOTIFICATION_TYPES = (
    "model_health_warning", "feature_drift_alert", "retraining_recommended",
    "retraining_started", "retraining_completed", "candidate_rejected",
    "candidate_promoted", "model_performance_improved", "model_performance_degraded",
)

_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


@dataclass(slots=True)
class Notification:
    """One structured notification -- the Agentic AI (or any other
    consumer) reads ``type``/``severity``/``message``/``payload``."""

    type: str
    severity: str
    message: str
    model_name: str
    timestamp: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type, "severity": self.severity, "message": self.message,
            "model_name": self.model_name, "timestamp": self.timestamp, "payload": self.payload,
        }


class NotificationManager:
    """Builds, records, and (subject to policy) dispatches notifications."""

    def __init__(self, policy: Optional[NotificationPolicy] = None) -> None:
        self.policy = policy or NotificationPolicy()
        self.history: List[Notification] = []
        self._handlers: List[Callable[[Notification], None]] = []

    def add_handler(self, handler: Callable[[Notification], None]) -> None:
        self._handlers.append(handler)

    def notify(
        self, type_: str, *, severity: str, message: str, model_name: str,
        timestamp: Optional[str] = None, **payload: Any,
    ) -> Notification:
        if type_ not in NOTIFICATION_TYPES:
            raise ValueError(f"Unknown notification type {type_!r}, expected one of {NOTIFICATION_TYPES}.")
        if severity not in _SEVERITY_ORDER:
            raise ValueError(f"severity={severity!r}, expected one of {tuple(_SEVERITY_ORDER)}.")
        note = Notification(
            type=type_, severity=severity, message=message, model_name=model_name,
            timestamp=timestamp or utc_timestamp(), payload=payload,
        )
        self.history.append(note)
        if _SEVERITY_ORDER[severity] >= _SEVERITY_ORDER[self.policy.min_severity]:
            for handler in self._handlers:
                handler(note)
        return note

    # ------------------------------------------------------------------ #
    # Convenience wrappers -- one per spec example notification.
    # ------------------------------------------------------------------ #
    def notify_model_health_warning(self, *, model_name: str, health_score: float, status: str) -> Notification:
        severity = "critical" if status == "critical" else "warning"
        return self.notify(
            "model_health_warning", severity=severity,
            message=f"{model_name}: health score {health_score:.1f} ({status}).",
            model_name=model_name, health_score=health_score, status=status,
        )

    def notify_feature_drift_alert(self, *, model_name: str, severity_label: str, overall_severity: float) -> Notification:
        severity = "critical" if severity_label == "severe" else "warning"
        return self.notify(
            "feature_drift_alert", severity=severity,
            message=f"{model_name}: feature drift {severity_label} (severity={overall_severity:.2f}).",
            model_name=model_name, severity_label=severity_label, overall_severity=overall_severity,
        )

    def notify_retraining_recommended(self, *, model_name: str, priority: str, reasons: List[str]) -> Notification:
        return self.notify(
            "retraining_recommended", severity="warning" if priority in ("low", "medium") else "critical",
            message=f"{model_name}: retraining recommended (priority={priority}).",
            model_name=model_name, priority=priority, reasons=list(reasons),
        )

    def notify_retraining_started(self, *, model_name: str) -> Notification:
        return self.notify("retraining_started", severity="info", message=f"{model_name}: retraining started.", model_name=model_name)

    def notify_retraining_completed(self, *, model_name: str, candidate_version: str) -> Notification:
        return self.notify(
            "retraining_completed", severity="info",
            message=f"{model_name}: retraining completed -- candidate {candidate_version}.",
            model_name=model_name, candidate_version=candidate_version,
        )

    def notify_candidate_rejected(self, *, model_name: str, version: str, reasons: List[str]) -> Notification:
        return self.notify(
            "candidate_rejected", severity="warning",
            message=f"{model_name}: candidate {version} rejected.", model_name=model_name,
            version=version, reasons=list(reasons),
        )

    def notify_candidate_promoted(self, *, model_name: str, version: str, reasons: List[str]) -> Notification:
        return self.notify(
            "candidate_promoted", severity="info",
            message=f"{model_name}: candidate {version} promoted to production.", model_name=model_name,
            version=version, reasons=list(reasons),
        )

    def notify_performance_improved(self, *, model_name: str, relative_improvement: float) -> Notification:
        return self.notify(
            "model_performance_improved", severity="info",
            message=f"{model_name}: performance improved by {relative_improvement * 100:+.2f}%.",
            model_name=model_name, relative_improvement=relative_improvement,
        )

    def notify_performance_degraded(self, *, model_name: str, relative_change: float) -> Notification:
        return self.notify(
            "model_performance_degraded", severity="warning",
            message=f"{model_name}: performance degraded by {relative_change * 100:+.2f}%.",
            model_name=model_name, relative_change=relative_change,
        )
