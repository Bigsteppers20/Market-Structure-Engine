"""Model registry for regression models.

``RegressionModelRegistry`` subclasses ``training.registry.ModelRegistry``
directly rather than reimplementing file-based storage: ``register()``,
``list_models()``, ``list_versions()``, and index management are all
inherited unchanged (``register()`` only ever calls ``.to_dict()``,
``.model_name``, ``.version`` on whatever it's given, so it works verbatim
for the richer metadata type below). Only ``get()`` is overridden, to
reconstruct :class:`RegressionModelMetadata` instead of the generic
``training.ModelMetadata`` -- and since ``all_metadata()`` calls
``self.get(...)`` internally, it picks up the override automatically with
no further changes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from training.registry import ModelRegistry


def _safe(value: str) -> str:
    """Filesystem-safe filename fragment -- mirrors training.registry's
    private helper of the same name (not imported directly since leading-
    underscore names aren't part of that module's public contract)."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in value)


@dataclass(slots=True)
class RegressionModelMetadata:
    """Everything the spec requires to be recorded for a registered
    regression model -- ``training.ModelMetadata``'s fields plus the two
    regression-specific ones (``regression_target``, ``prediction_horizon``)."""

    model_name: str
    version: str
    training_date: str
    training_dataset: str
    feature_version: str
    regression_target: str
    prediction_horizon: int
    performance_metrics: Dict[str, Any]
    supported_symbols: List[str]
    supported_timeframes: List[str]
    artifact_dir: str
    model_type: str = "linear"
    experiment_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegressionModelMetadata":
        return cls(**d)

    @property
    def key(self) -> str:
        return f"{self.model_name}@{self.version}"


class RegressionModelRegistry(ModelRegistry):
    """``training.ModelRegistry``, specialized to store
    :class:`RegressionModelMetadata` instead of the generic ``ModelMetadata``."""

    def get(self, model_name: str, version: Optional[str] = None) -> RegressionModelMetadata:
        index = self._load_index()
        versions = index.get(model_name)
        if not versions:
            raise KeyError(f"No registered regression model named {model_name!r}.")
        if version is None:
            version = versions[-1]
        elif version not in versions:
            raise KeyError(f"Model {model_name!r} has no version {version!r}. Known: {versions}")
        path = self.root / f"{_safe(model_name)}__{_safe(version)}.json"
        return RegressionModelMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))
