"""Versioning for classification models.

Builds on ``training.versioning.VersionInfo`` (engine/dataset/pipeline
versions, reused unmodified) by adding the identifiers a classification
model specifically needs to declare: its ordered class set and the horizon
it was labeled/trained for. A model trained for ``(SELL, NO_TRADE, BUY)`` at
a 5-candle horizon must never silently serve predictions for a different
class set or horizon -- that mismatch is caught by
:mod:`logistic_regression.validator`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

from training.versioning import VersionInfo, current_version_info

#: Version of this engine's own code (bump on any behavior change that could
#: affect historical predictions).
LOGISTIC_REGRESSION_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ClassificationModelVersion:
    """Full version identity of one trained classification model."""

    version_info: VersionInfo
    engine_version: str
    classes: Tuple[str, ...]
    prediction_horizon: int
    model_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_info": self.version_info.to_dict(),
            "engine_version": self.engine_version,
            "classes": list(self.classes),
            "prediction_horizon": self.prediction_horizon,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassificationModelVersion":
        return cls(
            version_info=VersionInfo.from_dict(d["version_info"]),
            engine_version=d["engine_version"],
            classes=tuple(d["classes"]),
            prediction_horizon=int(d["prediction_horizon"]),
            model_version=d.get("model_version", "1.0.0"),
        )


def current_classification_version(
    classes: Tuple[str, ...], prediction_horizon: int,
    feature_version: str = "1.0.0", model_version: str = "1.0.0",
) -> ClassificationModelVersion:
    """Build a :class:`ClassificationModelVersion` from the versions installed right now."""
    return ClassificationModelVersion(
        version_info=current_version_info(feature_version),
        engine_version=LOGISTIC_REGRESSION_ENGINE_VERSION,
        classes=tuple(classes),
        prediction_horizon=prediction_horizon,
        model_version=model_version,
    )
