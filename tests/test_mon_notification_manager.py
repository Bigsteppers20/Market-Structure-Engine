"""Tests for model_monitor.notification_manager -- structured notification
generation and handler dispatch (NOTIFICATION SYSTEM section)."""
from __future__ import annotations

import pytest

from model_monitor.config import NotificationPolicy
from model_monitor.notification_manager import NOTIFICATION_TYPES, NotificationManager


def test_rejects_unknown_type() -> None:
    mgr = NotificationManager()
    with pytest.raises(ValueError):
        mgr.notify("bogus_type", severity="info", message="x", model_name="m")


def test_rejects_unknown_severity() -> None:
    mgr = NotificationManager()
    with pytest.raises(ValueError):
        mgr.notify("model_health_warning", severity="bogus", message="x", model_name="m")


def test_notify_records_history() -> None:
    mgr = NotificationManager()
    mgr.notify("retraining_started", severity="info", message="started", model_name="m")
    assert len(mgr.history) == 1
    assert mgr.history[0].type == "retraining_started"


def test_handlers_invoked_above_min_severity() -> None:
    received = []
    mgr = NotificationManager(NotificationPolicy(min_severity="warning"))
    mgr.add_handler(received.append)
    mgr.notify("retraining_started", severity="info", message="x", model_name="m")
    mgr.notify("candidate_rejected", severity="warning", message="y", model_name="m")
    assert len(received) == 1
    assert received[0].type == "candidate_rejected"


def test_all_notification_types_have_convenience_wrappers() -> None:
    mgr = NotificationManager()
    mgr.notify_model_health_warning(model_name="m", health_score=42.0, status="critical")
    mgr.notify_feature_drift_alert(model_name="m", severity_label="severe", overall_severity=0.8)
    mgr.notify_retraining_recommended(model_name="m", priority="high", reasons=["reason"])
    mgr.notify_retraining_started(model_name="m")
    mgr.notify_retraining_completed(model_name="m", candidate_version="2")
    mgr.notify_candidate_rejected(model_name="m", version="2", reasons=["worse"])
    mgr.notify_candidate_promoted(model_name="m", version="2", reasons=["better"])
    mgr.notify_performance_improved(model_name="m", relative_improvement=0.1)
    mgr.notify_performance_degraded(model_name="m", relative_change=-0.1)
    types_emitted = {n.type for n in mgr.history}
    assert types_emitted == set(NOTIFICATION_TYPES)


def test_health_warning_severity_matches_status() -> None:
    mgr = NotificationManager()
    warning = mgr.notify_model_health_warning(model_name="m", health_score=55.0, status="warning")
    critical = mgr.notify_model_health_warning(model_name="m", health_score=20.0, status="critical")
    assert warning.severity == "warning"
    assert critical.severity == "critical"


def test_notification_to_dict_contains_payload() -> None:
    mgr = NotificationManager()
    note = mgr.notify_retraining_completed(model_name="m", candidate_version="7")
    d = note.to_dict()
    assert d["type"] == "retraining_completed"
    assert d["payload"]["candidate_version"] == "7"
    assert d["model_name"] == "m"


def test_timestamp_defaults_when_not_supplied() -> None:
    mgr = NotificationManager()
    note = mgr.notify("retraining_started", severity="info", message="x", model_name="m")
    assert note.timestamp  # non-empty ISO string
