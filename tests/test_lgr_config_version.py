"""Tests for logistic_regression.config and logistic_regression.version."""
from __future__ import annotations

import pytest

from logistic_regression.config import ClassificationConfig, DEFAULT_CLASSES
from logistic_regression.exceptions import (
    InvalidHorizonError,
    InvalidThresholdError,
    UnsupportedBalancingStrategyError,
    UnsupportedClassSetError,
)
from logistic_regression.version import (
    LOGISTIC_REGRESSION_ENGINE_VERSION,
    ClassificationModelVersion,
    current_classification_version,
)
from training.config import TrainingConfig


def test_defaults() -> None:
    cfg = ClassificationConfig()
    assert cfg.classes == DEFAULT_CLASSES == ("SELL", "NO_TRADE", "BUY")
    assert cfg.prediction_horizon == 5
    assert cfg.class_balancing == "none"
    assert cfg.calibration_method == "none"
    assert cfg.threshold_strategy == "argmax"
    assert isinstance(cfg.training_config, TrainingConfig)


def test_rejects_too_few_classes() -> None:
    with pytest.raises(UnsupportedClassSetError):
        ClassificationConfig(classes=("BUY",))


def test_rejects_duplicate_classes() -> None:
    with pytest.raises(UnsupportedClassSetError):
        ClassificationConfig(classes=("BUY", "SELL", "BUY"))


def test_rejects_invalid_horizon() -> None:
    with pytest.raises(InvalidHorizonError):
        ClassificationConfig(prediction_horizon=0)


def test_rejects_unknown_balancing_strategy() -> None:
    with pytest.raises(UnsupportedBalancingStrategyError):
        ClassificationConfig(class_balancing="bogus")


def test_rejects_unknown_calibration_method() -> None:
    with pytest.raises(InvalidThresholdError):
        ClassificationConfig(calibration_method="bogus")


def test_rejects_unknown_threshold_strategy() -> None:
    with pytest.raises(InvalidThresholdError):
        ClassificationConfig(threshold_strategy="bogus")


def test_rejects_negative_bootstrap() -> None:
    with pytest.raises(ValueError):
        ClassificationConfig(n_bootstrap=-1)


def test_extended_class_set_is_supported_without_api_change() -> None:
    """The architecture must support future class sets (STRONG_BUY, WEAK_BUY,
    EXIT_LONG, ...) without changing ClassificationConfig's shape."""
    cfg = ClassificationConfig(
        classes=("STRONG_SELL", "WEAK_SELL", "NO_TRADE", "WEAK_BUY", "STRONG_BUY", "EXIT_LONG", "EXIT_SHORT")
    )
    assert len(cfg.classes) == 7


def test_to_dict_from_dict_round_trip() -> None:
    cfg = ClassificationConfig(
        classes=("SELL", "NO_TRADE", "BUY"), prediction_horizon=10, min_pip_movement=8.0,
        class_balancing="class_weight", calibration_method="platt", threshold_strategy="custom",
        custom_thresholds={"BUY": 0.6, "SELL": 0.6, "NO_TRADE": 0.3},
    )
    restored = ClassificationConfig.from_dict(cfg.to_dict())
    assert restored.classes == ("SELL", "NO_TRADE", "BUY")
    assert restored.prediction_horizon == 10
    assert restored.min_pip_movement == 8.0
    assert restored.class_balancing == "class_weight"
    assert restored.calibration_method == "platt"
    assert restored.custom_thresholds == {"BUY": 0.6, "SELL": 0.6, "NO_TRADE": 0.3}
    assert isinstance(restored.training_config, TrainingConfig)


def test_current_classification_version_fields() -> None:
    v = current_classification_version(("SELL", "NO_TRADE", "BUY"), 5, feature_version="2.0.0")
    assert v.classes == ("SELL", "NO_TRADE", "BUY")
    assert v.prediction_horizon == 5
    assert v.engine_version == LOGISTIC_REGRESSION_ENGINE_VERSION
    assert v.version_info.feature_version == "2.0.0"


def test_version_dict_round_trip() -> None:
    v = current_classification_version(("SELL", "NO_TRADE", "BUY"), 5)
    restored = ClassificationModelVersion.from_dict(v.to_dict())
    assert restored == v
