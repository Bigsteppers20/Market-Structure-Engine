"""Tests for model_monitor.model_registry.ModelLifecycleRegistry --
lifecycle status transitions, counters, and MODEL AGING."""
from __future__ import annotations

import pytest

from model_monitor.model_registry import ModelLifecycleMetadata, ModelLifecycleRegistry
from training.registry import ModelRegistry


def _meta(**overrides):
    base = dict(
        model_name="m", version="1", task_type="regression", status="candidate",
        training_date="2026-01-01T00:00:00+00:00", training_dataset_size=1000,
        feature_version="1.0.0", training_version="1.0.0", strategy_version="1.0.0", dataset_version="1.0.0",
    )
    base.update(overrides)
    return ModelLifecycleMetadata(**base)


def test_is_a_training_model_registry_subclass() -> None:
    assert issubclass(ModelLifecycleRegistry, ModelRegistry)


def test_rejects_unknown_status() -> None:
    with pytest.raises(ValueError):
        _meta(status="bogus")


def test_model_age_days() -> None:
    meta = _meta(training_date="2026-01-01T00:00:00+00:00")
    assert meta.model_age_days("2026-01-11T00:00:00+00:00") == pytest.approx(10.0)


def test_model_age_never_negative() -> None:
    meta = _meta(training_date="2026-01-11T00:00:00+00:00")
    assert meta.model_age_days("2026-01-01T00:00:00+00:00") == 0.0


def test_prediction_accuracy_rate_none_when_no_predictions() -> None:
    meta = _meta()
    assert meta.prediction_accuracy_rate() is None


def test_prediction_accuracy_rate_computed() -> None:
    meta = _meta(live_prediction_count=10, correct_prediction_count=7)
    assert meta.prediction_accuracy_rate() == pytest.approx(0.7)


def test_register_and_get(tmp_path) -> None:
    reg = ModelLifecycleRegistry(tmp_path)
    reg.register(_meta())
    fetched = reg.get("m")
    assert isinstance(fetched, ModelLifecycleMetadata)
    assert fetched.status == "candidate"


def test_record_prediction_increments_counter(tmp_path) -> None:
    reg = ModelLifecycleRegistry(tmp_path)
    reg.register(_meta())
    reg.record_prediction("m", "1")
    reg.record_prediction("m", "1")
    assert reg.get("m").live_prediction_count == 2


def test_record_trade_outcome_increments_counters(tmp_path) -> None:
    reg = ModelLifecycleRegistry(tmp_path)
    reg.register(_meta())
    reg.record_trade_outcome("m", "1", correct=True)
    reg.record_trade_outcome("m", "1", correct=False)
    meta = reg.get("m")
    assert meta.completed_trade_count == 2
    assert meta.correct_prediction_count == 1


def test_promote_marks_production_and_archives_previous(tmp_path) -> None:
    reg = ModelLifecycleRegistry(tmp_path)
    reg.register(_meta(version="1", status="production"))
    reg.register(_meta(version="2", status="candidate"))
    reg.promote("m", "2", promotion_timestamp="2026-02-01T00:00:00+00:00")

    v1 = reg.get("m", "1")
    v2 = reg.get("m", "2")
    assert v1.status == "archived"
    assert v2.status == "production"
    assert v2.promotion_timestamp == "2026-02-01T00:00:00+00:00"


def test_archive(tmp_path) -> None:
    reg = ModelLifecycleRegistry(tmp_path)
    reg.register(_meta(version="1", status="candidate"))
    reg.archive("m", "1")
    assert reg.get("m", "1").status == "archived"


def test_production_version_returns_none_when_no_production(tmp_path) -> None:
    reg = ModelLifecycleRegistry(tmp_path)
    reg.register(_meta(version="1", status="candidate"))
    assert reg.production_version("m") is None


def test_production_version_finds_the_promoted_entry(tmp_path) -> None:
    reg = ModelLifecycleRegistry(tmp_path)
    reg.register(_meta(version="1", status="production"))
    reg.register(_meta(version="2", status="candidate"))
    found = reg.production_version("m")
    assert found is not None and found.version == "1"


def test_retraining_safety_never_leaves_two_production_versions(tmp_path) -> None:
    """The core RETRAINING SAFETY guarantee: promoting a new version must
    never leave the old one also marked production."""
    reg = ModelLifecycleRegistry(tmp_path)
    reg.register(_meta(version="1", status="production"))
    reg.register(_meta(version="2", status="candidate"))
    reg.promote("m", "2", promotion_timestamp="t")
    statuses = [reg.get("m", v).status for v in reg.list_versions("m")]
    assert statuses.count("production") == 1


def test_metadata_to_dict_from_dict_round_trip() -> None:
    meta = _meta(supported_symbols=["EUR_USD"], supported_timeframes=["M5"])
    restored = ModelLifecycleMetadata.from_dict(meta.to_dict())
    assert restored == meta
    assert restored.key == "m@1"
