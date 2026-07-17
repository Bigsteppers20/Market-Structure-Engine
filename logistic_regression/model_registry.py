"""Model registry for classification models.

``ClassificationModelRegistry`` subclasses ``training.registry.ModelRegistry``
directly rather than reimplementing file-based storage -- ``register()``,
``list_models()``, ``list_versions()``, and index management are all
inherited unchanged; only ``get()`` is overridden to reconstruct
:class:`ClassificationModelMetadata` instead of the generic
``training.ModelMetadata`` (and ``all_metadata()`` picks up the override
automatically since it calls ``self.get(...)`` internally).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from training.registry import ModelRegistry


def _safe(value: str) -> str:
    """Filesystem-safe filename fragment (independently defined -- see
    linear_regression.model_registry for the identical rationale: avoid
    coupling to training.registry's private helper of the same name)."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in value)


@dataclass(slots=True)
class ClassificationModelMetadata:
    """Everything the spec requires to be recorded for a registered
    classification model."""

    model_name: str
    version: str
    training_date: str
    feature_version: str
    training_dataset: str
    classification_labels: Tuple[str, ...]
    prediction_horizon: int
    calibration_method: str
    performance_metrics: Dict[str, Any]
    supported_symbols: List[str]
    supported_timeframes: List[str]
    artifact_dir: str
    experiment_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["classification_labels"] = list(self.classification_labels)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassificationModelMetadata":
        d = dict(d)
        d["classification_labels"] = tuple(d["classification_labels"])
        return cls(**d)

    @property
    def key(self) -> str:
        return f"{self.model_name}@{self.version}"


class ClassificationModelRegistry(ModelRegistry):
    """``training.ModelRegistry``, specialized to store
    :class:`ClassificationModelMetadata` instead of the generic ``ModelMetadata``."""

    def get(self, model_name: str, version: Optional[str] = None) -> ClassificationModelMetadata:
        index = self._load_index()
        versions = index.get(model_name)
        if not versions:
            raise KeyError(f"No registered classification model named {model_name!r}.")
        if version is None:
            version = versions[-1]
        elif version not in versions:
            raise KeyError(f"Model {model_name!r} has no version {version!r}. Known: {versions}")
        path = self.root / f"{_safe(model_name)}__{_safe(version)}.json"
        return ClassificationModelMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))
