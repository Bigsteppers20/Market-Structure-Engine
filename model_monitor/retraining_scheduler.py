"""Retraining mode dispatch (RETRAINING MODES section).

* **Manual** -- never auto-triggers; the caller is expected to only ever
  surface :class:`~model_monitor.retraining_policy.RetrainingRecommendation`
  as a notification.
* **Scheduled** -- triggers purely on elapsed time since the model was last
  trained (``lifecycle.training_date``), independent of health/drift.
* **Adaptive** -- triggers only when **all** of: health score below
  threshold, feature drift above threshold, enough new data exists, and
  the policy explicitly permits automatic retraining
  (``config.auto_retraining_enabled``) -- the exact AND-gate the spec's
  RETRAINING MODES section specifies (deliberately stricter than the
  general, OR-based :class:`RetrainingRecommendation`, which fires on any
  *one* condition to at least warrant a notification).
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .config import MonitorConfig, RetrainingScheduleConfig
    from .drift_detector import DriftReport
    from .health_engine import ModelHealthReport

RETRAINING_MODES = ("manual", "scheduled", "adaptive")


def is_due(schedule: "RetrainingScheduleConfig", last_trained_iso: str, now_iso: str) -> bool:
    """Whether ``schedule.interval_days`` have elapsed since ``last_trained_iso``."""
    elapsed_days = (datetime.fromisoformat(now_iso) - datetime.fromisoformat(last_trained_iso)).total_seconds() / 86400.0
    return elapsed_days >= schedule.interval_days


def should_trigger_retraining(
    *, mode: str, schedule: "RetrainingScheduleConfig", health_report: "ModelHealthReport",
    drift_report: "DriftReport", config: "MonitorConfig", last_trained_iso: str, now_iso: str,
    new_sample_count: int,
) -> Tuple[bool, str]:
    """Decide whether to actually run a retraining cycle right now, per
    ``mode``. Returns ``(triggered, reason)``."""
    if mode not in RETRAINING_MODES:
        raise ValueError(f"mode={mode!r}, expected one of {RETRAINING_MODES}.")

    if mode == "manual":
        return False, "Manual mode -- retraining is never automatic; surface the recommendation as a notification only."

    if mode == "scheduled":
        due = is_due(schedule, last_trained_iso, now_iso)
        if due:
            return True, f"Scheduled retraining is due ({schedule.frequency}, interval={schedule.interval_days:.1f} days)."
        return False, f"Scheduled retraining not yet due ({schedule.frequency}, interval={schedule.interval_days:.1f} days)."

    # adaptive -- strict AND-gate, per spec.
    if not config.auto_retraining_enabled:
        return False, "Adaptive mode requested, but auto_retraining_enabled=False -- policy does not permit automatic retraining."

    health_below = health_report.health.overall < config.health_threshold
    drift_above = drift_report.feature_drift.overall_severity > config.feature_drift_threshold
    enough_data = new_sample_count >= config.min_new_samples

    if health_below and drift_above and enough_data:
        return True, (
            f"Adaptive conditions met: health={health_report.health.overall:.1f} < {config.health_threshold:.1f}, "
            f"feature_drift={drift_report.feature_drift.overall_severity:.2f} > {config.feature_drift_threshold:.2f}, "
            f"new_samples={new_sample_count} >= {config.min_new_samples}."
        )

    unmet = []
    if not health_below:
        unmet.append("health score not below threshold")
    if not drift_above:
        unmet.append("feature drift not above threshold")
    if not enough_data:
        unmet.append(f"insufficient new data ({new_sample_count} < {config.min_new_samples})")
    return False, "Adaptive conditions not met: " + "; ".join(unmet) + "."
