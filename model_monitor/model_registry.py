"""Model lifecycle registry for the monitoring system.

Distinct from ``linear_regression.RegressionModelRegistry``/
``logistic_regression.ClassificationModelRegistry`` (which track *training*
metadata for their own engine): this registry tracks *monitoring/lifecycle*
state layered on top -- status (production/candidate/archived), model age,
live prediction/trade/correct-prediction counters (MODEL AGING section) --
keyed by the same ``model_name@version`` convention so it composes with,
rather than duplicates, those registries. Subclasses
``training.registry.ModelRegistry`` directly, same extension pattern used
by every other engine on this platform.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from training.registry import ModelRegistry

STATUSES = ("candidate", "production", "archived")


@dataclass(slots=True)
class ModelLifecycleMetadata:
    """Everything the MODEL AGING and VERSIONING sections require to be
    tracked for one monitored model version."""

    model_name: str
    version: str
    task_type: str
    """``"regression"`` or ``"classification"`` -- the only thing this
    registry ever branches on, purely for reporting (never for scoring
    logic, which lives in performance_monitor.py/health_score.py)."""
    status: str
    training_date: str
    training_dataset_size: int
    feature_version: str
    training_version: str
    strategy_version: str
    dataset_version: str
    supported_symbols: List[str] = field(default_factory=list)
    supported_timeframes: List[str] = field(default_factory=list)
    live_prediction_count: int = 0
    completed_trade_count: int = 0
    correct_prediction_count: int = 0
    promotion_timestamp: Optional[str] = None
    artifact_dir: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status={self.status!r}, expected one of {STATUSES}.")

    def model_age_days(self, now_iso: str) -> float:
        """Age in days between ``training_date`` and ``now_iso`` (both
        ISO-8601 timestamps, as produced by ``training.utils.utc_timestamp``)."""
        trained = datetime.fromisoformat(self.training_date)
        now = datetime.fromisoformat(now_iso)
        return max(0.0, (now - trained).total_seconds() / 86400.0)

    def prediction_accuracy_rate(self) -> Optional[float]:
        """``correct_prediction_count / live_prediction_count``, or
        ``None`` if no predictions have been logged yet."""
        if self.live_prediction_count == 0:
            return None
        return self.correct_prediction_count / self.live_prediction_count

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelLifecycleMetadata":
        return cls(**d)

    @property
    def key(self) -> str:
        return f"{self.model_name}@{self.version}"


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in value)


class ModelLifecycleRegistry(ModelRegistry):
    """``training.ModelRegistry``, specialized to store
    :class:`ModelLifecycleMetadata` instead of the generic ``ModelMetadata``,
    plus the mutation helpers MODEL AGING/promotion require (counters and
    status transitions are not something the base, training-time-only
    registry needs)."""

    def get(self, model_name: str, version: Optional[str] = None) -> ModelLifecycleMetadata:
        index = self._load_index()
        versions = index.get(model_name)
        if not versions:
            raise KeyError(f"No registered lifecycle entry named {model_name!r}.")
        if version is None:
            version = versions[-1]
        elif version not in versions:
            raise KeyError(f"Model {model_name!r} has no version {version!r}. Known: {versions}")
        path = self.root / f"{_safe(model_name)}__{_safe(version)}.json"
        return ModelLifecycleMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ #
    def record_prediction(self, model_name: str, version: str) -> ModelLifecycleMetadata:
        meta = self.get(model_name, version)
        meta.live_prediction_count += 1
        self.register(meta)
        return meta

    def record_trade_outcome(self, model_name: str, version: str, *, correct: bool) -> ModelLifecycleMetadata:
        meta = self.get(model_name, version)
        meta.completed_trade_count += 1
        if correct:
            meta.correct_prediction_count += 1
        self.register(meta)
        return meta

    def promote(self, model_name: str, version: str, *, promotion_timestamp: str) -> ModelLifecycleMetadata:
        """Mark ``version`` as production, archiving whatever was
        previously production for this ``model_name`` (retraining safety:
        the old production model is archived, never overwritten/deleted)."""
        for existing_version in self.list_versions(model_name):
            existing = self.get(model_name, existing_version)
            if existing.status == "production" and existing_version != version:
                existing.status = "archived"
                self.register(existing)
        meta = self.get(model_name, version)
        meta.status = "production"
        meta.promotion_timestamp = promotion_timestamp
        self.register(meta)
        return meta

    def archive(self, model_name: str, version: str) -> ModelLifecycleMetadata:
        meta = self.get(model_name, version)
        meta.status = "archived"
        self.register(meta)
        return meta

    def production_version(self, model_name: str) -> Optional[ModelLifecycleMetadata]:
        for version in reversed(self.list_versions(model_name)):
            meta = self.get(model_name, version)
            if meta.status == "production":
                return meta
        return None
