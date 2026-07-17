"""Pre-inference validation.

Reuses ``training.versioning`` for everything version/schema related
(feature schema version, feature count, feature ordering, dataset version,
training version) -- this module only adds the two checks specific to a
regression model: does it predict the target the caller asked for, and was
it trained for the requested horizon.
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

from .exceptions import TargetMismatchError
from .version import RegressionModelVersion


def validate_target_and_horizon(
    model_version: RegressionModelVersion, requested_target: Optional[str] = None,
    requested_horizon: Optional[int] = None,
) -> None:
    if requested_target is not None and requested_target != model_version.regression_target:
        raise TargetMismatchError(
            f"This model was trained for target {model_version.regression_target!r}, "
            f"but {requested_target!r} was requested."
        )
    if requested_horizon is not None and requested_horizon != model_version.prediction_horizon:
        raise TargetMismatchError(
            f"This model was trained for a {model_version.prediction_horizon}-candle horizon, "
            f"but a {requested_horizon}-candle horizon was requested."
        )


def validate_for_inference(
    *, model_version_info: VersionInfo, current_version_info: VersionInfo,
    feature_schema: FeatureSchema, feature_names: Sequence[str], X: np.ndarray,
    strict: bool = True,
) -> List[str]:
    """Full pre-inference check. Raises under ``strict=True``; returns a
    list of issue strings (empty = clean) under ``strict=False``."""
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
