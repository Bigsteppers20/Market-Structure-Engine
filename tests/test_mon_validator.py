"""Tests for model_monitor.validator -- snapshot self-consistency and
version/schema validation (VERSION VALIDATION coverage)."""
from __future__ import annotations

import numpy as np
import pytest

from model_monitor.exceptions import InvalidConfigError, UnknownModelError, VersionMismatchError
from model_monitor.model_registry import ModelLifecycleMetadata
from model_monitor.prediction_monitor import PredictionSnapshot
from model_monitor.validator import (
    assert_valid_prediction_snapshot,
    validate_for_monitoring,
    validate_lifecycle_identity,
    validate_prediction_snapshot,
)
from training.versioning import FeatureSchema, current_version_info


def _snapshot(**overrides) -> PredictionSnapshot:
    base = dict(
        task_type="regression", model_name="m", model_version="1", feature_version="1",
        training_version="1", symbol="EUR_USD", timeframe="M5", prediction_horizon=5,
        timestamp="t", decision_index=10, feature_vector=[1.0, 2.0], feature_names=["a", "b"],
        confidence=70.0, predicted_value=1.1, raw_predictions={},
    )
    base.update(overrides)
    return PredictionSnapshot(**base)


def test_valid_regression_snapshot_has_no_issues() -> None:
    assert validate_prediction_snapshot(_snapshot()) == []


def test_valid_classification_snapshot_has_no_issues() -> None:
    snap = _snapshot(
        task_type="classification", predicted_value=None, predicted_class="BUY",
        class_probabilities={"SELL": 0.2, "NO_TRADE": 0.3, "BUY": 0.5},
    )
    assert validate_prediction_snapshot(snap) == []


def test_unknown_task_type_flagged() -> None:
    issues = validate_prediction_snapshot(_snapshot(task_type="bogus"))
    assert any("task_type" in i for i in issues)


def test_feature_length_mismatch_flagged() -> None:
    issues = validate_prediction_snapshot(_snapshot(feature_vector=[1.0, 2.0, 3.0]))
    assert any("feature_vector length" in i for i in issues)


def test_confidence_out_of_range_flagged() -> None:
    issues = validate_prediction_snapshot(_snapshot(confidence=150.0))
    assert any("confidence" in i for i in issues)


def test_invalid_horizon_flagged() -> None:
    issues = validate_prediction_snapshot(_snapshot(prediction_horizon=0))
    assert any("prediction_horizon" in i for i in issues)


def test_regression_missing_predicted_value_flagged() -> None:
    issues = validate_prediction_snapshot(_snapshot(predicted_value=None))
    assert any("predicted_value" in i for i in issues)


def test_classification_missing_predicted_class_flagged() -> None:
    snap = _snapshot(task_type="classification", predicted_value=None, class_probabilities={"BUY": 1.0})
    issues = validate_prediction_snapshot(snap)
    assert any("predicted_class" in i for i in issues)


def test_classification_probabilities_must_sum_to_one() -> None:
    snap = _snapshot(
        task_type="classification", predicted_value=None, predicted_class="BUY",
        class_probabilities={"SELL": 0.5, "BUY": 0.9},
    )
    issues = validate_prediction_snapshot(snap)
    assert any("sum to" in i for i in issues)


def test_assert_valid_raises_on_bad_snapshot() -> None:
    with pytest.raises(InvalidConfigError):
        assert_valid_prediction_snapshot(_snapshot(confidence=-5.0))


def test_assert_valid_passes_silently_on_good_snapshot() -> None:
    assert_valid_prediction_snapshot(_snapshot())  # must not raise


# --------------------------------------------------------------------------- #
# validate_for_monitoring
# --------------------------------------------------------------------------- #
def test_validate_for_monitoring_strict_raises_on_version_mismatch() -> None:
    v1 = current_version_info("1.0.0")
    v2 = current_version_info("2.0.0")
    schema = FeatureSchema.from_feature_names(["a", "b"], v1)
    with pytest.raises(VersionMismatchError):
        validate_for_monitoring(
            model_version_info=v1, current_version_info=v2, feature_schema=schema,
            feature_names=["a", "b"], X=np.zeros((1, 2)), strict=True,
        )


def test_validate_for_monitoring_non_strict_collects_issues() -> None:
    v1 = current_version_info("1.0.0")
    v2 = current_version_info("2.0.0")
    schema = FeatureSchema.from_feature_names(["a", "b"], v1)
    issues = validate_for_monitoring(
        model_version_info=v1, current_version_info=v2, feature_schema=schema,
        feature_names=["a", "b"], X=np.zeros((1, 2)), strict=False,
    )
    assert issues


def test_validate_for_monitoring_clean_case() -> None:
    v1 = current_version_info("1.0.0")
    schema = FeatureSchema.from_feature_names(["a", "b"], v1)
    issues = validate_for_monitoring(
        model_version_info=v1, current_version_info=v1, feature_schema=schema,
        feature_names=["a", "b"], X=np.zeros((1, 2)), strict=False,
    )
    assert issues == []


# --------------------------------------------------------------------------- #
# validate_lifecycle_identity
# --------------------------------------------------------------------------- #
def _lifecycle() -> ModelLifecycleMetadata:
    return ModelLifecycleMetadata(
        model_name="m", version="1", task_type="regression", status="production",
        training_date="2026-01-01T00:00:00+00:00", training_dataset_size=100,
        feature_version="1", training_version="1", strategy_version="1", dataset_version="1",
    )


def test_validate_lifecycle_identity_passes_for_matching() -> None:
    validate_lifecycle_identity(_lifecycle(), expected_model_name="m", expected_version="1")


def test_validate_lifecycle_identity_rejects_name_mismatch() -> None:
    with pytest.raises(UnknownModelError):
        validate_lifecycle_identity(_lifecycle(), expected_model_name="other")


def test_validate_lifecycle_identity_rejects_version_mismatch() -> None:
    with pytest.raises(UnknownModelError):
        validate_lifecycle_identity(_lifecycle(), expected_model_name="m", expected_version="99")
