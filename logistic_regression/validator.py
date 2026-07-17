"""Pre-inference validation.

Reuses ``training.versioning`` for everything version/schema related
(feature schema version, feature count, feature ordering, dataset version,
training version, model version) -- this module only adds the checks
specific to a classifier: does it predict the class set the caller expects,
and was it trained for the requested horizon.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from training.versioning import (
    FeatureSchema,
    VersionInfo,
    assert_schema_compatible,
    validate_schema,
    verify_version_compatibility,
)

from .exceptions import ClassMismatchError
from .version import ClassificationModelVersion


def validate_classes_and_horizon(
    model_version: ClassificationModelVersion, requested_classes: Optional[Tuple[str, ...]] = None,
    requested_horizon: Optional[int] = None,
) -> None:
    if requested_classes is not None and tuple(requested_classes) != tuple(model_version.classes):
        raise ClassMismatchError(
            f"This model was trained for classes {model_version.classes!r}, "
            f"but {tuple(requested_classes)!r} was requested."
        )
    if requested_horizon is not None and requested_horizon != model_version.prediction_horizon:
        raise ClassMismatchError(
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
