"""Tests for training.artifacts."""
from __future__ import annotations

import numpy as np
import pytest

from training.artifacts import ArtifactManager, ModelPlaceholder


def test_save_and_load_joblib_round_trip(tmp_path) -> None:
    mgr = ArtifactManager(tmp_path)
    obj = {"a": np.array([1.0, 2.0, 3.0])}
    mgr.save_joblib(obj, "scaler")
    loaded = mgr.load_joblib("scaler")
    np.testing.assert_array_equal(loaded["a"], obj["a"])


def test_save_and_load_pickle_round_trip(tmp_path) -> None:
    mgr = ArtifactManager(tmp_path)
    obj = ModelPlaceholder(model_family="test_family", task_type="regression")
    mgr.save_pickle(obj, "model_placeholder")
    loaded = mgr.load_pickle("model_placeholder")
    assert loaded.model_family == "test_family"


def test_save_and_load_json_round_trip(tmp_path) -> None:
    mgr = ArtifactManager(tmp_path)
    obj = {"experiment_id": "exp_123", "count": 185}
    mgr.save_json(obj, "metadata")
    loaded = mgr.load_json("metadata")
    assert loaded == obj


def test_load_missing_artifact_raises(tmp_path) -> None:
    mgr = ArtifactManager(tmp_path)
    with pytest.raises(FileNotFoundError):
        mgr.load_json("metadata")
    with pytest.raises(FileNotFoundError):
        mgr.load_joblib("scaler")


def test_unknown_artifact_name_raises(tmp_path) -> None:
    mgr = ArtifactManager(tmp_path)
    with pytest.raises(ValueError):
        mgr.path_for("not_a_real_artifact")


def test_exists(tmp_path) -> None:
    mgr = ArtifactManager(tmp_path)
    assert not mgr.exists("config")
    mgr.save_json({"x": 1}, "config")
    assert mgr.exists("config")


def test_save_bundle_and_load_bundle_round_trip(tmp_path) -> None:
    mgr = ArtifactManager(tmp_path)
    placeholder = ModelPlaceholder(model_family="fam", task_type="classification")
    saved_paths = mgr.save_bundle(
        scaler={"mean_": 1.0},
        config={"scaler": "standard"},
        metadata={"experiment_id": "exp_x"},
        feature_schema={"feature_count": 3},
        model_placeholder=placeholder,
    )
    assert set(saved_paths) == {"scaler", "config", "metadata", "feature_schema", "model_placeholder"}
    for path in saved_paths.values():
        assert path.exists()

    loaded = mgr.load_bundle()
    assert loaded["config"]["scaler"] == "standard"
    assert loaded["metadata"]["experiment_id"] == "exp_x"
    assert loaded["model_placeholder"].model_family == "fam"
    assert "feature_selector" not in loaded  # never saved -> never loaded


def test_save_bundle_prefers_real_model_over_placeholder(tmp_path) -> None:
    mgr = ArtifactManager(tmp_path)
    saved = mgr.save_bundle(model={"coef": [1, 2, 3]}, model_placeholder=ModelPlaceholder("fam", "regression"))
    assert "model" in saved
    assert "model_placeholder" not in saved
