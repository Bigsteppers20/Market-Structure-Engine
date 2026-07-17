"""Tests for training.registry."""
from __future__ import annotations

import pytest

from training.registry import ModelMetadata, ModelRegistry


def _meta(**overrides) -> ModelMetadata:
    base = dict(
        model_name="trend_model", version="1.0.0", training_date="2026-01-01T00:00:00Z",
        feature_count=185, feature_schema_version="1.0.0", training_dataset_version="1.0.0",
        performance_metrics={"accuracy": 0.6}, supported_timeframes=["M5"],
        supported_symbols=["EUR_USD"], training_strategy="trend_following",
        artifact_dir="/tmp/artifacts/exp1",
    )
    base.update(overrides)
    return ModelMetadata(**base)


def test_register_and_get(tmp_path) -> None:
    reg = ModelRegistry(tmp_path)
    reg.register(_meta())
    fetched = reg.get("trend_model")
    assert fetched.version == "1.0.0"
    assert fetched.feature_count == 185
    assert fetched.performance_metrics == {"accuracy": 0.6}


def test_get_unknown_model_raises(tmp_path) -> None:
    reg = ModelRegistry(tmp_path)
    with pytest.raises(KeyError):
        reg.get("does_not_exist")


def test_get_unknown_version_raises(tmp_path) -> None:
    reg = ModelRegistry(tmp_path)
    reg.register(_meta())
    with pytest.raises(KeyError):
        reg.get("trend_model", version="9.9.9")


def test_multiple_versions_latest_by_default(tmp_path) -> None:
    reg = ModelRegistry(tmp_path)
    reg.register(_meta(version="1.0.0"))
    reg.register(_meta(version="2.0.0"))
    assert reg.get("trend_model").version == "2.0.0"  # latest registered
    assert reg.get("trend_model", version="1.0.0").version == "1.0.0"


def test_list_models_and_versions(tmp_path) -> None:
    reg = ModelRegistry(tmp_path)
    reg.register(_meta(model_name="model_a", version="1.0.0"))
    reg.register(_meta(model_name="model_b", version="1.0.0"))
    reg.register(_meta(model_name="model_a", version="2.0.0"))
    assert set(reg.list_models()) == {"model_a", "model_b"}
    assert reg.list_versions("model_a") == ["1.0.0", "2.0.0"]


def test_reregistering_same_version_overwrites_not_duplicates(tmp_path) -> None:
    reg = ModelRegistry(tmp_path)
    reg.register(_meta(version="1.0.0", performance_metrics={"accuracy": 0.5}))
    reg.register(_meta(version="1.0.0", performance_metrics={"accuracy": 0.9}))
    assert reg.list_versions("trend_model") == ["1.0.0"]
    assert reg.get("trend_model").performance_metrics == {"accuracy": 0.9}


def test_all_metadata_returns_every_entry(tmp_path) -> None:
    reg = ModelRegistry(tmp_path)
    reg.register(_meta(model_name="model_a", version="1.0.0"))
    reg.register(_meta(model_name="model_b", version="1.0.0"))
    all_meta = reg.all_metadata()
    assert {m.key for m in all_meta} == {"model_a@1.0.0", "model_b@1.0.0"}


def test_registry_persists_across_instances(tmp_path) -> None:
    ModelRegistry(tmp_path).register(_meta())
    reg2 = ModelRegistry(tmp_path)  # fresh instance, same root
    assert reg2.get("trend_model").version == "1.0.0"
