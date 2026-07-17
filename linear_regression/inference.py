"""Inference infrastructure for one trained regression target.

Wraps ``training.InferencePipeline`` (reused unmodified for artifact
loading, version verification, and feature-vector preparation via the fitted
``FeaturePipeline``/``FeatureScaler``/``FeatureSelector``) and adds the one
thing that package deliberately omits: loading and running the actual
fitted model. ``training.InferencePipeline`` never calls ``predict()`` by
design (it implements no model); this module is where that finally happens,
since predicting *is* this engine's job.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from training.artifacts import ArtifactManager
from training.inference import InferencePipeline
from training.versioning import VersionInfo

from .exceptions import ModelNotTrainedError
from .regression_model import RegressionModel
from .version import LINEAR_REGRESSION_ENGINE_VERSION, RegressionModelVersion


class RegressionInferencePipeline:
    """Loads one target's artifacts, validates compatibility, and predicts.

    Parameters mirror ``training.InferencePipeline`` -- see that module for
    the version/schema enforcement semantics (``strict`` raises immediately
    on any mismatch; non-strict collects warnings instead).
    """

    def __init__(self, artifact_dir: str | Path, feature_version: str = "1.0.0", strict: bool = True) -> None:
        self.artifact_dir = Path(artifact_dir)
        self._base = InferencePipeline(self.artifact_dir, feature_version=feature_version, strict=strict)
        self._manager = ArtifactManager(self.artifact_dir)
        self.model: Optional[RegressionModel] = None
        self.model_version: Optional[RegressionModelVersion] = None
        self._loaded = False

    def load(self) -> "RegressionInferencePipeline":
        self._base.load()
        if not self._manager.exists("model"):
            raise ModelNotTrainedError(
                f"No trained model found at {self.artifact_dir} -- was this trained with "
                "LinearRegressionTrainer (not the base training.Trainer)?"
            )
        self.model = self._manager.load_joblib("model")

        metadata = self._manager.load_json("metadata")
        config = self._manager.load_json("config") if self._manager.exists("config") else {}
        assert self._base.feature_schema is not None
        self.model_version = RegressionModelVersion(
            version_info=self._base.feature_schema.version_info,
            engine_version=LINEAR_REGRESSION_ENGINE_VERSION,
            regression_target=metadata.get("target_name", "unknown"),
            prediction_horizon=int(config.get("prediction_horizon", 0)),
        )
        self._loaded = True
        return self

    @property
    def version_warnings(self) -> List[str]:
        return self._base.version_warnings

    def predict(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(point_estimate, ensemble_std)`` for one target, from a
        raw ``(1, n_features)`` vector in the Market Structure Engine's own
        feature order."""
        if not self._loaded or self.model is None:
            raise ModelNotTrainedError("Call load() before predict().")
        X_prepared, _ = self._base.prepare(X, feature_names)
        point, std = self.model.predict_with_uncertainty(X_prepared)
        return point, std
