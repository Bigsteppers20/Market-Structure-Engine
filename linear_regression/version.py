"""Versioning for regression models.

Builds on ``training.versioning.VersionInfo`` (engine/dataset/pipeline
versions, reused unmodified) by adding the two extra identifiers a
regression model specifically needs to declare: which target it predicts
and what horizon it was trained for. A model trained on ``next_close`` at a
5-candle horizon must never silently serve predictions requested for
``future_volatility`` at a 20-candle horizon -- that mismatch is caught by
:mod:`linear_regression.validator`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from training.versioning import VersionInfo, current_version_info

#: Version of this engine's own code (bump on any behavior change that could
#: affect historical predictions -- e.g. a change to target_generator.py's
#: formulas or confidence.py's weighting).
LINEAR_REGRESSION_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class RegressionModelVersion:
    """Full version identity of one trained regression model."""

    version_info: VersionInfo
    engine_version: str
    regression_target: str
    prediction_horizon: int
    model_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_info": self.version_info.to_dict(),
            "engine_version": self.engine_version,
            "regression_target": self.regression_target,
            "prediction_horizon": self.prediction_horizon,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegressionModelVersion":
        return cls(
            version_info=VersionInfo.from_dict(d["version_info"]),
            engine_version=d["engine_version"],
            regression_target=d["regression_target"],
            prediction_horizon=int(d["prediction_horizon"]),
            model_version=d.get("model_version", "1.0.0"),
        )


def current_regression_version(
    regression_target: str, prediction_horizon: int,
    feature_version: str = "1.0.0", model_version: str = "1.0.0",
) -> RegressionModelVersion:
    """Build a :class:`RegressionModelVersion` from the versions installed right now."""
    return RegressionModelVersion(
        version_info=current_version_info(feature_version),
        engine_version=LINEAR_REGRESSION_ENGINE_VERSION,
        regression_target=regression_target,
        prediction_horizon=prediction_horizon,
        model_version=model_version,
    )
