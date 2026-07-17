"""Local model registry: tracks every trained model's metadata and where
its artifacts live, independent of any specific ML framework.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import ensure_dir, utc_timestamp


@dataclass(slots=True)
class ModelMetadata:
    """Everything the spec requires to be recorded for a registered model."""

    model_name: str
    version: str
    training_date: str
    feature_count: int
    feature_schema_version: str
    training_dataset_version: str
    performance_metrics: Dict[str, Any]
    supported_timeframes: List[str]
    supported_symbols: List[str]
    training_strategy: str
    artifact_dir: str
    model_family: str = "unspecified"
    task_type: str = "unspecified"
    experiment_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelMetadata":
        return cls(**d)

    @property
    def key(self) -> str:
        return f"{self.model_name}@{self.version}"


class ModelRegistry:
    """Flat, file-based registry: ``root/models/index.json`` plus one JSON
    per ``model_name@version`` entry."""

    def __init__(self, root: str | Path) -> None:
        self.root = ensure_dir(Path(root) / "models")
        self._index_path = self.root / "index.json"

    def register(self, metadata: ModelMetadata) -> Path:
        """Add (or overwrite) a model entry. Registration is idempotent per
        ``model_name@version`` -- registering the same key again replaces it."""
        path = self.root / f"{_safe(metadata.model_name)}__{_safe(metadata.version)}.json"
        path.write_text(json.dumps(metadata.to_dict(), indent=2, default=str), encoding="utf-8")
        self._update_index(metadata)
        return path

    def get(self, model_name: str, version: Optional[str] = None) -> ModelMetadata:
        """Fetch a specific version, or the most-recently-registered one if
        ``version`` is ``None``."""
        index = self._load_index()
        versions = index.get(model_name)
        if not versions:
            raise KeyError(f"No registered model named {model_name!r}.")
        if version is None:
            version = versions[-1]  # most recently appended
        elif version not in versions:
            raise KeyError(f"Model {model_name!r} has no version {version!r}. Known: {versions}")
        path = self.root / f"{_safe(model_name)}__{_safe(version)}.json"
        return ModelMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_models(self) -> List[str]:
        return sorted(self._load_index())

    def list_versions(self, model_name: str) -> List[str]:
        return list(self._load_index().get(model_name, []))

    def all_metadata(self) -> List[ModelMetadata]:
        return [
            self.get(name, version)
            for name, versions in self._load_index().items()
            for version in versions
        ]

    # ------------------------------------------------------------------ #
    def _load_index(self) -> Dict[str, List[str]]:
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def _update_index(self, metadata: ModelMetadata) -> None:
        index = self._load_index()
        versions = index.setdefault(metadata.model_name, [])
        if metadata.version in versions:
            versions.remove(metadata.version)
        versions.append(metadata.version)  # keep insertion order -> last = latest
        self._index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in value)
