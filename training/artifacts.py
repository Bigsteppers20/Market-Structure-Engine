"""Artifact persistence: scaler, feature selector, config, metadata,
reports, feature schema, and model (or its placeholder), all under one
directory per experiment.

Serialization uses joblib for Python objects (scaler/selector/pipeline --
consistent with how ``ml_pipeline.FeatureScaler`` already saves itself) and
plain JSON for everything text-shaped (config/metadata/reports/schema, so
they're human-readable and diffable). A raw-pickle path is also provided
per the spec's explicit "Joblib and Pickle" requirement.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from .utils import ensure_dir

ARTIFACT_FILENAMES: Dict[str, str] = {
    "scaler": "scaler.joblib",
    "feature_selector": "feature_selector.joblib",
    "feature_pipeline": "feature_pipeline.joblib",
    "config": "config.json",
    "metadata": "metadata.json",
    "training_report": "training_report.json",
    "evaluation_report": "evaluation_report.json",
    "feature_schema": "feature_schema.json",
    "model": "model.joblib",
    "model_placeholder": "model_placeholder.joblib",
}

JSON_ARTIFACTS = {"config", "metadata", "training_report", "evaluation_report", "feature_schema"}


@dataclass(slots=True)
class ModelPlaceholder:
    """Stands in for a trained model artifact.

    This infrastructure implements no machine learning algorithm, so there
    is never a real fitted model to serialize here -- a concrete
    :class:`training.trainer.Trainer` subclass (added later, elsewhere)
    would call :meth:`ArtifactManager.save_model` with its actual fitted
    model object instead of this placeholder, using the exact same
    joblib/pickle mechanism.
    """

    model_family: str
    task_type: str
    note: str = "No trained model -- this is infrastructure only, no ML algorithm is implemented."


class ArtifactManager:
    """Reads/writes every artifact type for one experiment's output directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = ensure_dir(Path(root))

    def path_for(self, name: str) -> Path:
        if name not in ARTIFACT_FILENAMES:
            raise ValueError(f"Unknown artifact name {name!r}. Known: {sorted(ARTIFACT_FILENAMES)}")
        return self.root / ARTIFACT_FILENAMES[name]

    # ------------------------------------------------------------------ #
    # generic joblib / pickle / json primitives
    # ------------------------------------------------------------------ #
    def save_joblib(self, obj: Any, name: str) -> Path:
        path = self.path_for(name)
        joblib.dump(obj, path)
        return path

    def load_joblib(self, name: str) -> Any:
        path = self.path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"Artifact {name!r} not found at {path}")
        return joblib.load(path)

    def save_pickle(self, obj: Any, name: str, suffix: str = ".pkl") -> Path:
        path = self.root / f"{Path(ARTIFACT_FILENAMES[name]).stem}{suffix}"
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def load_pickle(self, name: str, suffix: str = ".pkl") -> Any:
        path = self.root / f"{Path(ARTIFACT_FILENAMES[name]).stem}{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"Pickle artifact {name!r} not found at {path}")
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def save_json(self, obj: Dict[str, Any], name: str) -> Path:
        path = self.path_for(name)
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return path

    def load_json(self, name: str) -> Dict[str, Any]:
        path = self.path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"Artifact {name!r} not found at {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def exists(self, name: str) -> bool:
        return self.path_for(name).exists()

    # ------------------------------------------------------------------ #
    # bundle-level convenience
    # ------------------------------------------------------------------ #
    def save_bundle(
        self,
        *,
        scaler: Any = None,
        feature_selector: Any = None,
        feature_pipeline: Any = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        training_report: Optional[Dict[str, Any]] = None,
        evaluation_report: Optional[Dict[str, Any]] = None,
        feature_schema: Optional[Dict[str, Any]] = None,
        model: Any = None,
        model_placeholder: Optional[ModelPlaceholder] = None,
    ) -> Dict[str, Path]:
        """Save every provided artifact; skips any left as ``None``."""
        saved: Dict[str, Path] = {}
        joblib_objs = {
            "scaler": scaler, "feature_selector": feature_selector,
            "feature_pipeline": feature_pipeline,
        }
        for name, obj in joblib_objs.items():
            if obj is not None:
                saved[name] = self.save_joblib(obj, name)

        json_objs = {
            "config": config, "metadata": metadata, "training_report": training_report,
            "evaluation_report": evaluation_report, "feature_schema": feature_schema,
        }
        for name, obj in json_objs.items():
            if obj is not None:
                saved[name] = self.save_json(obj, name)

        if model is not None:
            saved["model"] = self.save_joblib(model, "model")
        elif model_placeholder is not None:
            saved["model_placeholder"] = self.save_joblib(model_placeholder, "model_placeholder")

        return saved

    def load_bundle(self) -> Dict[str, Any]:
        """Load every artifact present in this directory (skips missing ones)."""
        loaded: Dict[str, Any] = {}
        for name in ("scaler", "feature_selector", "feature_pipeline"):
            if self.exists(name):
                loaded[name] = self.load_joblib(name)
        for name in JSON_ARTIFACTS:
            if self.exists(name):
                loaded[name] = self.load_json(name)
        for name in ("model", "model_placeholder"):
            if self.exists(name):
                loaded[name] = self.load_joblib(name)
        return loaded
