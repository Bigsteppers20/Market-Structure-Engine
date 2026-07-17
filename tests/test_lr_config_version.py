"""Tests for linear_regression.config and linear_regression.version."""
from __future__ import annotations

import pytest

from linear_regression.config import RegressionConfig
from linear_regression.exceptions import InvalidHorizonError, UnsupportedModelTypeError, UnsupportedTargetError
from linear_regression.version import current_regression_version, LINEAR_REGRESSION_ENGINE_VERSION
from training.config import TrainingConfig


def test_defaults() -> None:
    cfg = RegressionConfig()
    assert cfg.targets == ["next_close"]
    assert cfg.prediction_horizon == 5
    assert cfg.model_type == "linear"
    assert isinstance(cfg.training_config, TrainingConfig)


def test_rejects_unsupported_model_type() -> None:
    with pytest.raises(UnsupportedModelTypeError):
        RegressionConfig(model_type="bogus")


def test_rejects_invalid_horizon() -> None:
    with pytest.raises(InvalidHorizonError):
        RegressionConfig(prediction_horizon=0)


def test_rejects_empty_targets() -> None:
    with pytest.raises(UnsupportedTargetError):
        RegressionConfig(targets=[])


def test_common_horizons_are_all_valid() -> None:
    for horizon in (1, 3, 5, 10, 20, 50):
        cfg = RegressionConfig(prediction_horizon=horizon)
        assert cfg.prediction_horizon == horizon


def test_to_dict_from_dict_round_trip() -> None:
    cfg = RegressionConfig(targets=["next_close", "next_high"], prediction_horizon=10, model_type="ridge",
                            model_hyperparameters={"alpha": 2.0})
    restored = RegressionConfig.from_dict(cfg.to_dict())
    assert restored.targets == ["next_close", "next_high"]
    assert restored.prediction_horizon == 10
    assert restored.model_hyperparameters == {"alpha": 2.0}
    assert isinstance(restored.training_config, TrainingConfig)


def test_current_regression_version_fields() -> None:
    v = current_regression_version("next_close", 5, feature_version="2.0.0")
    assert v.regression_target == "next_close"
    assert v.prediction_horizon == 5
    assert v.engine_version == LINEAR_REGRESSION_ENGINE_VERSION
    assert v.version_info.feature_version == "2.0.0"


def test_version_dict_round_trip() -> None:
    v = current_regression_version("next_close", 5)
    from linear_regression.version import RegressionModelVersion
    restored = RegressionModelVersion.from_dict(v.to_dict())
    assert restored == v
