"""Tests for training.versioning -- the version/schema compatibility gate."""
from __future__ import annotations

import numpy as np
import pytest

from training.versioning import (
    FeatureSchema,
    SchemaMismatchError,
    VersionInfo,
    VersionMismatchError,
    assert_schema_compatible,
    current_version_info,
    validate_schema,
    verify_version_compatibility,
)


def _vi(**overrides) -> VersionInfo:
    base = dict(
        feature_version="1.0.0", schema_version="1.0.0",
        market_structure_engine_version="1.0.0", dataset_builder_version="1.0.0",
        training_pipeline_version="1.0.0",
    )
    base.update(overrides)
    return VersionInfo(**base)


def test_current_version_info_matches_installed_packages() -> None:
    import market_structure
    import ml_pipeline
    vi = current_version_info("2.0.0")
    assert vi.feature_version == "2.0.0"
    assert vi.market_structure_engine_version == market_structure.__version__
    assert vi.dataset_builder_version == ml_pipeline.__version__


def test_verify_version_compatibility_passes_when_identical() -> None:
    v1, v2 = _vi(), _vi()
    verify_version_compatibility(v1, v2)  # must not raise


def test_verify_version_compatibility_raises_on_any_field_mismatch() -> None:
    v1 = _vi()
    for field_name in ("feature_version", "schema_version", "market_structure_engine_version",
                       "dataset_builder_version", "training_pipeline_version"):
        v2 = _vi(**{field_name: "9.9.9"})
        with pytest.raises(VersionMismatchError, match=field_name):
            verify_version_compatibility(v1, v2)


def test_version_info_dict_round_trip() -> None:
    v1 = _vi(feature_version="3.1.4")
    v2 = VersionInfo.from_dict(v1.to_dict())
    assert v1 == v2


def test_version_info_from_dict_rejects_missing_field() -> None:
    with pytest.raises(ValueError):
        VersionInfo.from_dict({"feature_version": "1.0.0"})


def test_feature_schema_fingerprint_and_count() -> None:
    schema = FeatureSchema.from_feature_names(["a", "b", "c"], _vi())
    assert schema.feature_count == 3
    assert schema.fingerprint == FeatureSchema.from_feature_names(["a", "b", "c"], _vi()).fingerprint
    assert schema.fingerprint != FeatureSchema.from_feature_names(["c", "b", "a"], _vi()).fingerprint


def test_feature_schema_rejects_inconsistent_count() -> None:
    with pytest.raises(ValueError):
        FeatureSchema(feature_names=["a", "b"], feature_count=5, version_info=_vi())


def test_feature_schema_dict_round_trip() -> None:
    schema = FeatureSchema.from_feature_names(["x", "y"], _vi())
    restored = FeatureSchema.from_dict(schema.to_dict())
    assert restored.feature_names == schema.feature_names
    assert restored.version_info == schema.version_info
    assert restored.fingerprint == schema.fingerprint


def test_validate_schema_clean() -> None:
    schema = FeatureSchema.from_feature_names(["a", "b", "c"], _vi())
    assert validate_schema(schema, ["a", "b", "c"]) == []


def test_validate_schema_detects_count_mismatch() -> None:
    schema = FeatureSchema.from_feature_names(["a", "b", "c"], _vi())
    issues = validate_schema(schema, ["a", "b"])
    assert any("count" in i for i in issues)


def test_validate_schema_detects_ordering_mismatch() -> None:
    schema = FeatureSchema.from_feature_names(["a", "b", "c"], _vi())
    issues = validate_schema(schema, ["c", "b", "a"])
    assert any("ordering" in i for i in issues)


def test_validate_schema_detects_missing_and_extra_names() -> None:
    schema = FeatureSchema.from_feature_names(["a", "b", "c"], _vi())
    issues = validate_schema(schema, ["a", "b", "z"])
    assert any("missing" in i for i in issues)
    assert any("unexpected" in i for i in issues)


def test_validate_schema_detects_dtype_mismatch() -> None:
    schema = FeatureSchema.from_feature_names(["a", "b"], _vi(), dtype="float64")
    X = np.zeros((3, 2), dtype=np.float32)
    issues = validate_schema(schema, ["a", "b"], X)
    assert any("dtype" in i for i in issues)


def test_assert_schema_compatible_raises_schema_mismatch_error() -> None:
    schema = FeatureSchema.from_feature_names(["a", "b", "c"], _vi())
    with pytest.raises(SchemaMismatchError):
        assert_schema_compatible(schema, ["a", "b"])


def test_schema_mismatch_error_is_a_version_mismatch_error() -> None:
    assert issubclass(SchemaMismatchError, VersionMismatchError)
