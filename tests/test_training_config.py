"""Tests for training.config."""
from __future__ import annotations

import pytest

from training.config import TrainingConfig


def test_defaults_are_valid() -> None:
    cfg = TrainingConfig()
    assert cfg.scaler == "standard"
    assert cfg.random_seed == 42
    assert cfg.feature_pipeline_config.impute_invalid is True  # from ml_pipeline.DatasetConfig default


def test_rejects_invalid_scaler() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(scaler="bogus")


def test_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(random_seed=-1)


def test_to_dict_and_from_dict_round_trip() -> None:
    cfg = TrainingConfig(experiment_name="my_exp", feature_version="2.0.0", scaler="robust",
                          feature_selector="kbest", feature_selector_kwargs={"k": 10},
                          random_seed=99, supported_symbols=["EUR_USD"], supported_timeframes=["M5"])
    d = cfg.to_dict()
    restored = TrainingConfig.from_dict(d)
    assert restored.experiment_name == "my_exp"
    assert restored.feature_version == "2.0.0"
    assert restored.scaler == "robust"
    assert restored.feature_selector_kwargs == {"k": 10}
    assert restored.random_seed == 99
    assert restored.supported_symbols == ["EUR_USD"]
    assert restored.feature_pipeline_config.window_size == cfg.feature_pipeline_config.window_size


def test_from_dict_rejects_unknown_field() -> None:
    d = TrainingConfig().to_dict()
    d["bogus_field"] = 1
    with pytest.raises(ValueError):
        TrainingConfig.from_dict(d)


def test_json_round_trip(tmp_path) -> None:
    cfg = TrainingConfig(experiment_name="json_test", strategy_version="4.2.0")
    path = cfg.to_json(tmp_path / "cfg.json")
    assert path.exists()
    restored = TrainingConfig.from_json(path)
    assert restored.experiment_name == "json_test"
    assert restored.strategy_version == "4.2.0"


def test_output_dir_property() -> None:
    cfg = TrainingConfig(output_root="some/output/dir")
    assert str(cfg.output_dir) in ("some/output/dir", "some\\output\\dir")
