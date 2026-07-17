"""Versioning for monitored models.

Builds on ``training.versioning.VersionInfo`` (engine/dataset/pipeline
versions, reused unmodified) by adding the identifiers the monitoring
system itself needs to track per the spec's VERSIONING section: which
strategy/dataset version a model was trained against, plus the training
and (if applicable) promotion timestamps. Deliberately independent of
``linear_regression.RegressionModelVersion``/``logistic_regression.ClassificationModelVersion``
-- this is a model-agnostic wrapper any task type can populate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from training.versioning import VersionInfo, current_version_info

#: Version of this system's own code (bump on any behavior change that
#: could affect historical health/drift/retraining decisions).
MODEL_MONITOR_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class MonitoringVersion:
    """Full version identity of one monitored model."""

    version_info: VersionInfo
    monitor_version: str
    model_version: str
    strategy_version: str
    dataset_version: str
    training_timestamp: str
    promotion_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_info": self.version_info.to_dict(),
            "monitor_version": self.monitor_version,
            "model_version": self.model_version,
            "strategy_version": self.strategy_version,
            "dataset_version": self.dataset_version,
            "training_timestamp": self.training_timestamp,
            "promotion_timestamp": self.promotion_timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MonitoringVersion":
        return cls(
            version_info=VersionInfo.from_dict(d["version_info"]),
            monitor_version=d["monitor_version"],
            model_version=d["model_version"],
            strategy_version=d["strategy_version"],
            dataset_version=d["dataset_version"],
            training_timestamp=d["training_timestamp"],
            promotion_timestamp=d.get("promotion_timestamp"),
        )

    def promoted(self, promotion_timestamp: str) -> "MonitoringVersion":
        """Return a copy stamped with a promotion timestamp -- this
        dataclass is frozen, so promotion never mutates history in place."""
        return MonitoringVersion(
            version_info=self.version_info, monitor_version=self.monitor_version,
            model_version=self.model_version, strategy_version=self.strategy_version,
            dataset_version=self.dataset_version, training_timestamp=self.training_timestamp,
            promotion_timestamp=promotion_timestamp,
        )


def current_monitoring_version(
    *, model_version: str, strategy_version: str, dataset_version: str,
    training_timestamp: str, feature_version: str = "1.0.0",
) -> MonitoringVersion:
    """Build a :class:`MonitoringVersion` from the versions installed right now."""
    return MonitoringVersion(
        version_info=current_version_info(feature_version),
        monitor_version=MODEL_MONITOR_VERSION,
        model_version=model_version, strategy_version=strategy_version,
        dataset_version=dataset_version, training_timestamp=training_timestamp,
    )
