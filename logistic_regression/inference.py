"""Inference infrastructure for the trained classifier.

Wraps ``training.InferencePipeline`` (reused unmodified for artifact
loading, version verification, and feature-vector preparation via the fitted
``FeaturePipeline``/``FeatureScaler``/``FeatureSelector``) and adds the one
thing that package deliberately omits: loading and running the actual
fitted model.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from training.artifacts import ArtifactManager
from training.inference import InferencePipeline

from .classification_model import ClassificationModel
from .exceptions import ModelNotTrainedError
from .version import LOGISTIC_REGRESSION_ENGINE_VERSION, ClassificationModelVersion


class ClassificationInferencePipeline:
    """Loads one trained classifier's artifacts, validates compatibility, and predicts."""

    def __init__(self, artifact_dir: str | Path, feature_version: str = "1.0.0", strict: bool = True) -> None:
        self.artifact_dir = Path(artifact_dir)
        self._base = InferencePipeline(self.artifact_dir, feature_version=feature_version, strict=strict)
        self._manager = ArtifactManager(self.artifact_dir)
        self.model: Optional[ClassificationModel] = None
        self.model_version: Optional[ClassificationModelVersion] = None
        self._loaded = False

    def load(self) -> "ClassificationInferencePipeline":
        self._base.load()
        if not self._manager.exists("model"):
            raise ModelNotTrainedError(
                f"No trained model found at {self.artifact_dir} -- was this trained with "
                "LogisticRegressionTrainer (not the base training.Trainer)?"
            )
        self.model = self._manager.load_joblib("model")

        config = self._manager.load_json("config") if self._manager.exists("config") else {}
        assert self._base.feature_schema is not None
        self.model_version = ClassificationModelVersion(
            version_info=self._base.feature_schema.version_info,
            engine_version=LOGISTIC_REGRESSION_ENGINE_VERSION,
            classes=tuple(config.get("classes", self.model.classes_)),
            prediction_horizon=int(config.get("prediction_horizon", 0)),
        )
        self._loaded = True
        return self

    @property
    def version_warnings(self) -> List[str]:
        return self._base.version_warnings

    def predict_proba(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(probabilities, bootstrap_agreement)`` for a raw
        ``(1, n_features)`` vector in the Market Structure Engine's own
        feature order. ``bootstrap_agreement`` is ``None`` if the model has
        no bootstrap ensemble."""
        if not self._loaded or self.model is None:
            raise ModelNotTrainedError("Call load() before predict_proba().")
        X_prepared, _ = self._base.prepare(X, feature_names)
        proba = self.model.predict_proba(X_prepared)
        agreement = self.model.bootstrap_agreement(X_prepared)
        return proba, agreement
