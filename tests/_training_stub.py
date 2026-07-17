"""Test-only stub Trainer subclasses.

These are NOT machine learning algorithms -- ``MeanBaselineStub`` predicts
the training-set mean and ``MajorityClassStub`` predicts the training-set
majority class, the simplest possible test doubles for exercising
``training.Trainer``'s orchestration logic (preprocessing, scaling,
selection, metric computation, artifact persistence, experiment logging,
registry registration) without depending on any real model library.
Not exported by ``training/__init__.py`` and not part of the shipped
infrastructure.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from training.trainer import Trainer


class MeanBaselineStub(Trainer):
    """Predicts the training-set mean for every row. Regression test double."""

    @property
    def model_family(self) -> str:
        return "mean_baseline_stub"

    @property
    def task_type(self) -> str:
        return "regression"

    def build_model(self, hyperparameters: Dict[str, Any]) -> Any:
        return {"offset": hyperparameters.get("offset", 0.0)}

    def fit_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        model["mean"] = float(np.mean(y_train)) + model["offset"]
        return model

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], model["mean"])

    def feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        return {name: 0.0 for name in feature_names[:3]}


class MajorityClassStub(Trainer):
    """Predicts the training-set majority class. Classification test double."""

    def __init__(self, config, n_classes: int = 3, repo_dir=None) -> None:
        super().__init__(config, repo_dir=repo_dir)
        self.n_classes = n_classes

    @property
    def model_family(self) -> str:
        return "majority_class_stub"

    @property
    def task_type(self) -> str:
        return "classification"

    def build_model(self, hyperparameters: Dict[str, Any]) -> Any:
        return {}

    def fit_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        values, counts = np.unique(y_train, return_counts=True)
        model["majority"] = int(values[np.argmax(counts)])
        return model

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], model["majority"])

    def predict_proba(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        proba = np.zeros((X.shape[0], self.n_classes))
        proba[:, model["majority"]] = 1.0
        return proba
