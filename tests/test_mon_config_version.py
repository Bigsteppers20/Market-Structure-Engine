"""Tests for model_monitor.config and model_monitor.version."""
from __future__ import annotations

import pytest

from model_monitor.config import (
    DEFAULT_HEALTH_WEIGHTS,
    MonitorConfig,
    NotificationPolicy,
    PromotionPolicy,
    RetrainingScheduleConfig,
)
from model_monitor.exceptions import InvalidConfigError
from model_monitor.version import MODEL_MONITOR_VERSION, MonitoringVersion, current_monitoring_version


def test_defaults() -> None:
    cfg = MonitorConfig()
    assert cfg.health_threshold == 60.0
    assert cfg.retraining_mode == "manual"
    assert cfg.health_weights == DEFAULT_HEALTH_WEIGHTS


def test_rejects_out_of_range_health_threshold() -> None:
    with pytest.raises(InvalidConfigError):
        MonitorConfig(health_threshold=150.0)


def test_rejects_out_of_range_feature_drift_threshold() -> None:
    with pytest.raises(InvalidConfigError):
        MonitorConfig(feature_drift_threshold=1.5)


def test_rejects_unknown_retraining_mode() -> None:
    with pytest.raises(InvalidConfigError):
        MonitorConfig(retraining_mode="bogus")


def test_rejects_incomplete_health_weights() -> None:
    with pytest.raises(InvalidConfigError):
        MonitorConfig(health_weights={"prediction_accuracy": 1.0})


def test_rejects_negative_min_new_samples() -> None:
    with pytest.raises(InvalidConfigError):
        MonitorConfig(min_new_samples=-1)


def test_to_dict_from_dict_round_trip() -> None:
    cfg = MonitorConfig(health_threshold=70.0, retraining_mode="adaptive", auto_retraining_enabled=True)
    restored = MonitorConfig.from_dict(cfg.to_dict())
    assert restored.health_threshold == 70.0
    assert restored.retraining_mode == "adaptive"
    assert isinstance(restored.retraining_schedule, RetrainingScheduleConfig)
    assert isinstance(restored.promotion_policy, PromotionPolicy)
    assert isinstance(restored.notification_policy, NotificationPolicy)


def test_promotion_policy_validation() -> None:
    with pytest.raises(InvalidConfigError):
        PromotionPolicy(min_relative_improvement=-0.1)
    with pytest.raises(InvalidConfigError):
        PromotionPolicy(tie_tolerance=-0.1)


def test_notification_policy_validation() -> None:
    with pytest.raises(InvalidConfigError):
        NotificationPolicy(min_severity="bogus")


def test_retraining_schedule_config_intervals() -> None:
    assert RetrainingScheduleConfig(frequency="daily").interval_days == 1.0
    assert RetrainingScheduleConfig(frequency="weekly").interval_days == 7.0
    assert RetrainingScheduleConfig(frequency="monthly").interval_days == 30.0
    assert RetrainingScheduleConfig(frequency="custom", custom_interval_days=3.5).interval_days == 3.5


def test_retraining_schedule_custom_requires_interval() -> None:
    with pytest.raises(InvalidConfigError):
        RetrainingScheduleConfig(frequency="custom")


def test_retraining_schedule_rejects_unknown_frequency() -> None:
    with pytest.raises(InvalidConfigError):
        RetrainingScheduleConfig(frequency="bogus")


def test_current_monitoring_version_fields() -> None:
    v = current_monitoring_version(
        model_version="1.0.0", strategy_version="2.0.0", dataset_version="3.0.0",
        training_timestamp="2026-01-01T00:00:00+00:00",
    )
    assert v.monitor_version == MODEL_MONITOR_VERSION
    assert v.strategy_version == "2.0.0"
    assert v.promotion_timestamp is None


def test_version_promoted_returns_new_frozen_copy() -> None:
    v = current_monitoring_version(
        model_version="1.0.0", strategy_version="2.0.0", dataset_version="3.0.0",
        training_timestamp="2026-01-01T00:00:00+00:00",
    )
    promoted = v.promoted("2026-02-01T00:00:00+00:00")
    assert v.promotion_timestamp is None
    assert promoted.promotion_timestamp == "2026-02-01T00:00:00+00:00"
    assert promoted.model_version == v.model_version


def test_version_dict_round_trip() -> None:
    v = current_monitoring_version(
        model_version="1.0.0", strategy_version="2.0.0", dataset_version="3.0.0",
        training_timestamp="2026-01-01T00:00:00+00:00",
    ).promoted("2026-02-01T00:00:00+00:00")
    restored = MonitoringVersion.from_dict(v.to_dict())
    assert restored == v
