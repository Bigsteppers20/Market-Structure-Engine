"""Tests for linear_regression.model_registry."""
from __future__ import annotations

import pytest

from linear_regression.model_registry import RegressionModelMetadata, RegressionModelRegistry
from training.registry import ModelRegistry


def _meta(**overrides):
    base = dict(
        model_name="lr_close", version="1.0.0", training_date="2026-01-01T00:00:00Z",
        training_dataset="1.0.0", feature_version="1.0.0", regression_target="next_close",
        prediction_horizon=5, performance_metrics={"r2": 0.4}, supported_symbols=["EUR_USD"],
        supported_timeframes=["M5"], artifact_dir="/tmp/artifacts/exp1",
    )
    base.update(overrides)
    return RegressionModelMetadata(**base)


def test_is_a_training_model_registry_subclass() -> None:
    assert issubclass(RegressionModelRegistry, ModelRegistry)


def test_register_and_get_richer_metadata(tmp_path) -> None:
    reg = RegressionModelRegistry(tmp_path)
    reg.register(_meta())
    fetched = reg.get("lr_close")
    assert isinstance(fetched, RegressionModelMetadata)
    assert fetched.regression_target == "next_close"
    assert fetched.prediction_horizon == 5


def test_get_unknown_raises(tmp_path) -> None:
    reg = RegressionModelRegistry(tmp_path)
    with pytest.raises(KeyError):
        reg.get("nope")


def test_all_metadata_uses_overridden_get(tmp_path) -> None:
    reg = RegressionModelRegistry(tmp_path)
    reg.register(_meta(model_name="a"))
    reg.register(_meta(model_name="b"))
    all_meta = reg.all_metadata()
    assert all(isinstance(m, RegressionModelMetadata) for m in all_meta)
    assert {m.model_name for m in all_meta} == {"a", "b"}


def test_list_models_and_versions_inherited_unchanged(tmp_path) -> None:
    reg = RegressionModelRegistry(tmp_path)
    reg.register(_meta(version="1.0.0"))
    reg.register(_meta(version="2.0.0"))
    assert reg.list_models() == ["lr_close"]
    assert reg.list_versions("lr_close") == ["1.0.0", "2.0.0"]
    assert reg.get("lr_close").version == "2.0.0"  # latest by default
