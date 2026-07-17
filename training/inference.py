"""Inference infrastructure: load a trained model's artifacts, verify it is
safe to use against the current code/feature versions, and prepare a raw
feature vector into the exact form the model was trained on.

This module stops at a fully processed, scaled, selected feature matrix --
it never calls ``model.predict()``. Passing that matrix to a real model is
the responsibility of whatever future model-specific code consumes this
pipeline's output.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .artifacts import ArtifactManager
from .versioning import (
    FeatureSchema,
    VersionInfo,
    assert_schema_compatible,
    current_version_info,
    validate_schema,
    verify_version_compatibility,
)


class InferencePipeline:
    """Loads one experiment's artifacts and prepares feature vectors for it.

    Parameters
    ----------
    artifact_dir:
        Directory written by ``Trainer.run()`` (contains ``feature_schema.json``,
        ``scaler.joblib``, etc.).
    feature_version:
        The feature-configuration version the caller expects to be running
        under -- compared against what the artifact was trained with.
    strict:
        When True (default), any version or schema mismatch raises
        immediately. When False, mismatches are collected in
        :attr:`version_warnings`/:attr:`schema_warnings` instead of raising --
        intended only for controlled migration testing, never production
        inference.
    """

    def __init__(self, artifact_dir: str | Path, feature_version: str = "1.0.0", strict: bool = True) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.feature_version = feature_version
        self.strict = strict
        self._manager = ArtifactManager(self.artifact_dir)

        self.feature_schema: Optional[FeatureSchema] = None
        self.current_version: Optional[VersionInfo] = None
        self.feature_pipeline = None
        self.scaler = None
        self.feature_selector = None
        self.version_warnings: List[str] = []
        self.schema_warnings: List[str] = []
        self._loaded = False

    # ------------------------------------------------------------------ #
    def load(self) -> "InferencePipeline":
        """Load artifacts and validate versions. Call once before :meth:`prepare`."""
        if not self._manager.exists("feature_schema"):
            raise FileNotFoundError(
                f"No feature_schema.json in {self.artifact_dir} -- cannot safely run "
                "inference without the schema this model was trained against."
            )
        self.feature_schema = FeatureSchema.from_dict(self._manager.load_json("feature_schema"))
        self.current_version = current_version_info(self.feature_version)

        if self.strict:
            verify_version_compatibility(self.feature_schema.version_info, self.current_version)
        else:
            try:
                verify_version_compatibility(self.feature_schema.version_info, self.current_version)
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: convert to a warning
                self.version_warnings.append(str(exc))

        self.feature_pipeline = self._manager.load_joblib("feature_pipeline") if self._manager.exists("feature_pipeline") else None
        self.scaler = self._manager.load_joblib("scaler") if self._manager.exists("scaler") else None
        self.feature_selector = self._manager.load_joblib("feature_selector") if self._manager.exists("feature_selector") else None
        self._loaded = True
        return self

    # ------------------------------------------------------------------ #
    def prepare(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Validate schema and apply the exact fit-time pipeline/scaler/selector.

        Returns ``(X_processed, feature_names_processed)``, ready to be
        passed to a future model's ``predict(X_processed)`` -- this method
        never calls one itself.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before prepare().")
        assert self.feature_schema is not None

        if self.strict:
            assert_schema_compatible(self.feature_schema, feature_names, X)
        else:
            self.schema_warnings = validate_schema(self.feature_schema, feature_names, X)

        names = list(feature_names)
        X_out = X
        if self.feature_pipeline is not None:
            X_out, names = self.feature_pipeline.transform(X_out, feature_names)
        if self.scaler is not None:
            X_out = self.scaler.transform(X_out)
        if self.feature_selector is not None:
            X_out, names = self.feature_selector.transform(X_out, names)
        return X_out, names
