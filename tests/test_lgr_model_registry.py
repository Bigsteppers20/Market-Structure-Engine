"""Tests for logistic_regression.model_registry."""
from __future__ import annotations

import pytest

from logistic_regression.model_registry import ClassificationModelMetadata, ClassificationModelRegistry
from training.registry import ModelRegistry


def _meta(**overrides):
    base = dict(
        model_name="lgr_eurusd", version="1.0.0", training_date="2026-01-01T00:00:00Z",
        feature_version="1.0.0", training_dataset="1.0.0",
        classification_labels=("SELL", "NO_TRADE", "BUY"), prediction_horizon=5,
        calibration_method="platt", performance_metrics={"accuracy": 0.6},
        supported_symbols=["EUR_USD"], supported_timeframes=["M5"],
        artifact_dir="/tmp/artifacts/exp1",
    )
    base.update(overrides)
    return ClassificationModelMetadata(**base)


def test_is_a_training_model_registry_subclass() -> None:
    assert issubclass(ClassificationModelRegistry, ModelRegistry)


def test_register_and_get_richer_metadata(tmp_path) -> None:
    reg = ClassificationModelRegistry(tmp_path)
    reg.register(_meta())
    fetched = reg.get("lgr_eurusd")
    assert isinstance(fetched, ClassificationModelMetadata)
    assert fetched.classification_labels == ("SELL", "NO_TRADE", "BUY")
    assert fetched.prediction_horizon == 5
    assert fetched.calibration_method == "platt"


def test_get_unknown_raises(tmp_path) -> None:
    reg = ClassificationModelRegistry(tmp_path)
    with pytest.raises(KeyError):
        reg.get("nope")


def test_get_unknown_version_raises(tmp_path) -> None:
    reg = ClassificationModelRegistry(tmp_path)
    reg.register(_meta(version="1.0.0"))
    with pytest.raises(KeyError):
        reg.get("lgr_eurusd", version="9.9.9")


def test_all_metadata_uses_overridden_get(tmp_path) -> None:
    reg = ClassificationModelRegistry(tmp_path)
    reg.register(_meta(model_name="a"))
    reg.register(_meta(model_name="b"))
    all_meta = reg.all_metadata()
    assert all(isinstance(m, ClassificationModelMetadata) for m in all_meta)
    assert {m.model_name for m in all_meta} == {"a", "b"}


def test_list_models_and_versions_inherited_unchanged(tmp_path) -> None:
    reg = ClassificationModelRegistry(tmp_path)
    reg.register(_meta(version="1.0.0"))
    reg.register(_meta(version="2.0.0"))
    assert reg.list_models() == ["lgr_eurusd"]
    assert reg.list_versions("lgr_eurusd") == ["1.0.0", "2.0.0"]
    assert reg.get("lgr_eurusd").version == "2.0.0"  # latest by default


def test_metadata_to_dict_from_dict_round_trip() -> None:
    meta = _meta()
    restored = ClassificationModelMetadata.from_dict(meta.to_dict())
    assert restored == meta
    assert restored.key == "lgr_eurusd@1.0.0"
