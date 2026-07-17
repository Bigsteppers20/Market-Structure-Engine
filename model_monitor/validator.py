"""Pre-monitoring validation.

Reuses ``training.versioning`` for feature schema/version compatibility
checks (same pattern as ``linear_regression.validator``/
``logistic_regression.validator``) -- this module only adds the checks
specific to monitoring: is a :class:`~model_monitor.prediction_monitor.PredictionSnapshot`
internally consistent, and is a lifecycle entry's version identity
consistent with what a caller expects.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from training.versioning import (
    FeatureSchema,
    VersionInfo,
    assert_schema_compatible,
    validate_schema,
    verify_version_compatibility,
)

from .exceptions import InvalidConfigError, UnknownModelError
from .model_registry import ModelLifecycleMetadata
from .prediction_monitor import PredictionSnapshot

TASK_TYPES = ("regression", "classification")


def validate_prediction_snapshot(snapshot: PredictionSnapshot) -> List[str]:
    """Structural self-consistency checks on one snapshot. Returns a list
    of issue strings (empty = clean) -- never raises, since a caller
    logging many predictions should be able to collect/skip bad ones
    rather than have the whole batch abort."""
    issues: List[str] = []
    if snapshot.task_type not in TASK_TYPES:
        issues.append(f"task_type={snapshot.task_type!r}, expected one of {TASK_TYPES}.")
    if len(snapshot.feature_vector) != len(snapshot.feature_names):
        issues.append(
            f"feature_vector length ({len(snapshot.feature_vector)}) != "
            f"feature_names length ({len(snapshot.feature_names)})."
        )
    if not (0.0 <= snapshot.confidence <= 100.0):
        issues.append(f"confidence={snapshot.confidence!r} outside [0, 100].")
    if snapshot.prediction_horizon < 1:
        issues.append(f"prediction_horizon={snapshot.prediction_horizon!r} must be >= 1.")
    if snapshot.task_type == "regression" and snapshot.predicted_value is None:
        issues.append("task_type='regression' but predicted_value is None.")
    if snapshot.task_type == "classification":
        if snapshot.predicted_class is None:
            issues.append("task_type='classification' but predicted_class is None.")
        if not snapshot.class_probabilities:
            issues.append("task_type='classification' but class_probabilities is empty.")
        elif abs(sum(snapshot.class_probabilities.values()) - 1.0) > 1e-6:
            issues.append(f"class_probabilities sum to {sum(snapshot.class_probabilities.values())!r}, expected 1.0.")
    return issues


def assert_valid_prediction_snapshot(snapshot: PredictionSnapshot) -> None:
    issues = validate_prediction_snapshot(snapshot)
    if issues:
        raise InvalidConfigError("Invalid PredictionSnapshot:\n  " + "\n  ".join(issues))


def validate_for_monitoring(
    *, model_version_info: VersionInfo, current_version_info: VersionInfo,
    feature_schema: FeatureSchema, feature_names: Sequence[str], X: Optional[np.ndarray] = None,
    strict: bool = True,
) -> List[str]:
    """Full pre-monitoring version/schema check -- same contract as
    ``logistic_regression.validator.validate_for_inference``: raises under
    ``strict=True``, returns a list of issue strings under ``strict=False``."""
    if strict:
        verify_version_compatibility(model_version_info, current_version_info)
        assert_schema_compatible(feature_schema, feature_names, X)
        return []
    issues: List[str] = []
    try:
        verify_version_compatibility(model_version_info, current_version_info)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: convert to a warning
        issues.append(str(exc))
    issues.extend(validate_schema(feature_schema, feature_names, X))
    return issues


def validate_lifecycle_identity(lifecycle: ModelLifecycleMetadata, *, expected_model_name: str, expected_version: Optional[str] = None) -> None:
    """Raise :class:`UnknownModelError` if a fetched lifecycle entry doesn't
    match what the caller expected (defends against a stale/mismatched
    registry lookup silently monitoring the wrong model)."""
    if lifecycle.model_name != expected_model_name:
        raise UnknownModelError(
            f"Expected lifecycle entry for model_name={expected_model_name!r}, got {lifecycle.model_name!r}."
        )
    if expected_version is not None and lifecycle.version != expected_version:
        raise UnknownModelError(
            f"Expected lifecycle entry for version={expected_version!r}, got {lifecycle.version!r}."
        )
