"""Feature/schema versioning and compatibility enforcement.

Every artifact bundle this package saves is stamped with a
:class:`VersionInfo` (five independently-tracked version numbers) and a
:class:`FeatureSchema` (exact feature names/order/dtype). Before any
inference, :func:`verify_version_compatibility` and :func:`validate_schema`
re-check both against the running environment and the incoming feature
vector, raising :class:`VersionMismatchError` /
:class:`SchemaMismatchError` (a subclass) the moment either would silently
produce wrong predictions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Sequence

import numpy as np

from .utils import hash_strings

#: Bump this whenever the *training pipeline itself* changes behavior in a
#: way that could affect a model trained with it (new preprocessing default,
#: new artifact layout, etc.).
TRAINING_PIPELINE_VERSION = "1.0.0"

#: Bump this whenever the *shape/meaning* of the feature schema this
#: infrastructure expects changes (independent of the engine's own version).
FEATURE_SCHEMA_VERSION = "1.0.0"


class VersionMismatchError(RuntimeError):
    """Raised when a model artifact's recorded versions don't match the
    versions of the code trying to use it for inference."""


class SchemaMismatchError(VersionMismatchError):
    """Raised when a feature vector's count/order/names/dtype don't match
    the schema a model artifact was trained against."""


@dataclass(frozen=True, slots=True)
class VersionInfo:
    """The five version numbers tracked for every trained model.

    Attributes
    ----------
    feature_version:
        Caller-supplied version of the *feature configuration* used
        (e.g. which ``EngineConfig``/``DatasetConfig`` preset). Not derived
        automatically -- the caller decides when this changes.
    schema_version:
        :data:`FEATURE_SCHEMA_VERSION` at build time.
    market_structure_engine_version:
        ``market_structure.__version__`` at build time.
    dataset_builder_version:
        ``ml_pipeline.__version__`` at build time.
    training_pipeline_version:
        :data:`TRAINING_PIPELINE_VERSION` at build time.
    """

    feature_version: str
    schema_version: str
    market_structure_engine_version: str
    dataset_builder_version: str
    training_pipeline_version: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VersionInfo":
        known = {f.name for f in fields(cls)}
        missing = known - set(d)
        if missing:
            raise ValueError(f"VersionInfo dict is missing field(s): {sorted(missing)}")
        return cls(**{k: d[k] for k in known})


def current_version_info(feature_version: str = "1.0.0") -> VersionInfo:
    """Build a :class:`VersionInfo` from the versions actually installed
    right now (imports ``market_structure``/``ml_pipeline`` lazily so this
    module has no hard import-order dependency on them)."""
    from market_structure import __version__ as mse_version
    from ml_pipeline import __version__ as builder_version

    return VersionInfo(
        feature_version=feature_version,
        schema_version=FEATURE_SCHEMA_VERSION,
        market_structure_engine_version=mse_version,
        dataset_builder_version=builder_version,
        training_pipeline_version=TRAINING_PIPELINE_VERSION,
    )


def verify_version_compatibility(expected: VersionInfo, actual: VersionInfo) -> None:
    """Raise :class:`VersionMismatchError` if any tracked version differs."""
    mismatches = [
        f"{f.name}: trained with {getattr(expected, f.name)!r}, "
        f"running {getattr(actual, f.name)!r}"
        for f in fields(VersionInfo)
        if getattr(expected, f.name) != getattr(actual, f.name)
    ]
    if mismatches:
        raise VersionMismatchError(
            "Version mismatch between the model artifact and the running "
            "environment:\n  " + "\n  ".join(mismatches)
        )


@dataclass(slots=True)
class FeatureSchema:
    """Exact feature contract a model was trained against.

    Attributes
    ----------
    feature_names:
        Ordered feature names, exactly as emitted by
        ``MarketStructureEngine.feature_vector()`` (optionally reduced by a
        ``FeatureSelector``).
    feature_count:
        ``len(feature_names)`` -- stored explicitly so a corrupted/truncated
        name list is itself detectable.
    dtype:
        Expected numpy dtype name for the feature matrix (``"float64"``).
    version_info:
        The :class:`VersionInfo` this schema was captured under.
    fingerprint:
        Stable hash of ``feature_names`` (order-sensitive) for a fast
        equality pre-check before doing a full field-by-field diff.
    """

    feature_names: List[str]
    feature_count: int
    version_info: VersionInfo
    dtype: str = "float64"
    fingerprint: str = field(default="")

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = hash_strings(self.feature_names)
        if self.feature_count != len(self.feature_names):
            raise ValueError(
                f"feature_count ({self.feature_count}) != len(feature_names) "
                f"({len(self.feature_names)})"
            )

    @classmethod
    def from_feature_names(
        cls, feature_names: Sequence[str], version_info: VersionInfo, dtype: str = "float64"
    ) -> "FeatureSchema":
        names = list(feature_names)
        return cls(feature_names=names, feature_count=len(names), version_info=version_info, dtype=dtype)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "feature_count": self.feature_count,
            "dtype": self.dtype,
            "fingerprint": self.fingerprint,
            "version_info": self.version_info.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSchema":
        return cls(
            feature_names=list(d["feature_names"]),
            feature_count=int(d["feature_count"]),
            dtype=d.get("dtype", "float64"),
            fingerprint=d.get("fingerprint", ""),
            version_info=VersionInfo.from_dict(d["version_info"]),
        )


def validate_schema(
    expected: FeatureSchema, feature_names: Sequence[str], X: np.ndarray | None = None
) -> List[str]:
    """Check feature count, ordering, names, and (optionally) dtype.

    Returns a list of human-readable mismatch descriptions -- empty means
    compatible. Does not raise; callers that need a hard failure should use
    :func:`assert_schema_compatible`.
    """
    issues: List[str] = []
    actual_names = list(feature_names)

    if len(actual_names) != expected.feature_count:
        issues.append(
            f"feature count mismatch: expected {expected.feature_count}, got {len(actual_names)}"
        )
    if actual_names != expected.feature_names:
        if sorted(actual_names) == sorted(expected.feature_names):
            issues.append("feature ordering mismatch: same names, different order")
        else:
            missing = set(expected.feature_names) - set(actual_names)
            extra = set(actual_names) - set(expected.feature_names)
            if missing:
                issues.append(f"missing feature(s): {sorted(missing)}")
            if extra:
                issues.append(f"unexpected feature(s): {sorted(extra)}")
    if X is not None and str(X.dtype) != expected.dtype:
        issues.append(f"dtype mismatch: expected {expected.dtype}, got {X.dtype}")
    return issues


def assert_schema_compatible(
    expected: FeatureSchema, feature_names: Sequence[str], X: np.ndarray | None = None
) -> None:
    """Raise :class:`SchemaMismatchError` if :func:`validate_schema` finds any issue."""
    issues = validate_schema(expected, feature_names, X)
    if issues:
        raise SchemaMismatchError("Feature schema is incompatible:\n  " + "\n  ".join(issues))
