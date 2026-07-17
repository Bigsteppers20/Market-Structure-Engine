"""``LogisticRegressionTrainer`` -- a concrete subclass of ``training.Trainer``.

Implements the four abstract hooks (``model_family``, ``task_type``,
``build_model``, ``fit_model``, ``predict``) plus the optional
``predict_proba``/``feature_importance`` hooks; every other responsibility
(dataset loading, ``FeaturePipeline``/``FeatureScaler``/``FeatureSelector``
orchestration, metric computation, evaluation-report generation, experiment
logging, model-registry registration) comes free from the unmodified base
class via ``Trainer.run()`` -- the same extension pattern proven by
``linear_regression.trainer.LinearRegressionTrainer``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from training.artifacts import ArtifactManager
from training.experiment import ExperimentRecord
from training.metrics import TrainingStatistics
from training.trainer import Trainer
from ml_pipeline.splitter import SplitResult
from ml_pipeline.dataset_builder import Dataset

from .classification_model import ClassificationModel
from .config import ClassificationConfig
from .evaluator import ClassificationEvaluator
from .metrics import _brier_macro, compute_all_classification_metrics
from .model_health import compute_model_health


class LogisticRegressionTrainer(Trainer):
    """Trains a multi-class logistic regression classifier."""

    #: Fraction of the (time-ordered) training split carved out as an
    #: internal, self-contained holdout purely to estimate historical
    #: out-of-sample balanced accuracy for the confidence engine -- distinct
    #: from ClassificationModel's own internal calibration holdout, so the
    #: "historical accuracy" stat is never contaminated by calibration data.
    INTERNAL_HOLDOUT_FRACTION = 0.15

    def __init__(self, classification_config: ClassificationConfig, repo_dir: Optional[Path] = None) -> None:
        super().__init__(classification_config.training_config, repo_dir=repo_dir)
        self.classification_config = classification_config
        self._model: Optional[ClassificationModel] = None

    # ------------------------------------------------------------------ #
    @property
    def model_family(self) -> str:
        return "logistic_regression"

    @property
    def task_type(self) -> str:
        return "classification"

    def build_model(self, hyperparameters: Dict[str, Any]) -> ClassificationModel:
        cfg = self.classification_config
        return ClassificationModel(
            class_balancing=cfg.class_balancing, calibration_method=cfg.calibration_method,
            n_bootstrap=cfg.n_bootstrap, random_state=self.config.random_seed, **hyperparameters,
        )

    def fit_model(self, model: ClassificationModel, X_train: np.ndarray, y_train: np.ndarray) -> ClassificationModel:
        cfg = self.classification_config
        y_train = np.asarray(y_train)

        n = X_train.shape[0]
        val_size = max(5, int(n * self.INTERNAL_HOLDOUT_FRACTION))
        historical_balanced_accuracy = 0.0
        have_holdout = n - val_size >= 10
        if have_holdout:
            X_fit, X_holdout = X_train[:-val_size], X_train[-val_size:]
            y_fit, y_holdout = y_train[:-val_size], y_train[-val_size:]
            if len(np.unique(y_fit)) >= 2:
                probe = ClassificationModel(
                    class_balancing=cfg.class_balancing, calibration_method="none",
                    n_bootstrap=0, random_state=self.config.random_seed,
                )
                probe.fit(X_fit, y_fit, list(cfg.classes))
                holdout_pred = np.argmax(probe.predict_proba(X_holdout), axis=1)
                historical_balanced_accuracy = float(balanced_accuracy_score(y_holdout, holdout_pred))

        model.fit(X_train, y_train, list(cfg.classes))

        # Confidence-supporting stats travel with the model artifact itself
        # (same pattern as linear_regression.trainer).
        model.historical_balanced_accuracy_ = historical_balanced_accuracy
        model.train_feature_mean_ = X_train.mean(axis=0)
        train_std = X_train.std(axis=0)
        model.train_feature_std_ = np.where(train_std < 1e-9, 1.0, train_std)
        if have_holdout:
            proba_holdout = model.predict_proba(X_holdout)
            model.calibration_error_ = _brier_macro(y_holdout, proba_holdout)
        else:
            model.calibration_error_ = None

        # Phase 8-style Decision Engine metadata (see model_health.py) --
        # a free blend of the two holdout stats just computed above.
        model.model_health_ = compute_model_health(model.historical_balanced_accuracy_, model.calibration_error_)

        self._model = model
        return model

    def predict(self, model: ClassificationModel, X: np.ndarray) -> np.ndarray:
        """Encoded-int predictions (matching dataset.y_cls's encoding) -- for
        real string class-name predictions, use predictor.py at inference time."""
        return np.argmax(model.predict_proba(X), axis=1)

    def predict_proba(self, model: ClassificationModel, X: np.ndarray) -> Optional[np.ndarray]:
        return model.predict_proba(X)

    def feature_importance(self, model: ClassificationModel, feature_names: List[str]) -> Optional[Dict[str, float]]:
        return model.feature_importance(feature_names)

    # ------------------------------------------------------------------ #
    def run(
        self, dataset: Dataset, split: SplitResult, *, target_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None, expected_schema=None,
    ) -> ExperimentRecord:
        """Identical to ``Trainer.run()`` (called via ``super()``, unmodified),
        plus three additive steps -- see
        ``linear_regression.trainer.LinearRegressionTrainer.run()`` for the
        identical rationale on the first two (persisting the real model +
        full config, since the base class only ever saves a placeholder and
        its own config). The third (richer evaluation report) is specific to
        this engine: the base ``Trainer.run()``'s own ``EvaluationEngine``
        pass only ever computes generic accuracy/precision/recall/f1 (its
        ``compute_fn`` dispatch never passes probabilities), so it cannot by
        itself satisfy the spec's calibration-curve/coefficient-diagnostics/
        feature-importance-report/prediction-and-probability-distribution
        requirements -- :class:`~logistic_regression.evaluator.ClassificationEvaluator`
        was built specifically to add those on top.

        ``target_name`` is accepted only for signature compatibility with
        the abstract base class's contract -- classification always trains
        against ``dataset.y_cls``, so it's derived automatically when omitted.
        """
        target_name = target_name or "_".join(self.classification_config.classes)
        hyperparameters = hyperparameters or self.classification_config.model_hyperparameters
        record = super().run(
            dataset, split, target_name=target_name, hyperparameters=hyperparameters,
            expected_schema=expected_schema,
        )
        assert self._model is not None
        manager = ArtifactManager(record.artifact_dir)
        manager.save_joblib(self._model, "model")
        manager.save_json(self.classification_config.to_dict(), "config")
        self._augment_evaluation_report(dataset, split, target_name, record, manager)
        return record

    # ------------------------------------------------------------------ #
    def _augment_evaluation_report(
        self, dataset: Dataset, split: SplitResult, target_name: str,
        record: ExperimentRecord, manager: ArtifactManager,
    ) -> None:
        """Reapply the already-fitted (never refit) feature pipeline/scaler/
        selector to the val/test splits so the richer, probability-aware
        report can be built -- ``Trainer.run()`` computes these internally
        but does not expose them, so this is the only way to get them
        without duplicating the base class's fitting logic."""
        model = self._model
        assert model is not None
        cfg = self.classification_config

        def _prepared(idx: np.ndarray) -> Optional[np.ndarray]:
            if idx.shape[0] == 0:
                return None
            X, names = dataset.X[idx], dataset.feature_names
            if self._fitted_pipeline is not None:
                X, names = self._fitted_pipeline.transform(X, names)
            if self._fitted_scaler is not None:
                X = self._fitted_scaler.transform(X)
            if self._fitted_selector is not None:
                X, names = self._fitted_selector.transform(X, names)
            return X

        X_val, X_test = _prepared(split.val_idx), _prepared(split.test_idx)
        y_val = dataset.y_cls[split.val_idx] if split.val_idx.shape[0] else None
        y_test = dataset.y_cls[split.test_idx] if split.test_idx.shape[0] else None

        y_val_pred = y_val_proba = y_test_pred = y_test_proba = None
        if X_val is not None:
            y_val_proba = model.predict_proba(X_val)
            y_val_pred = np.argmax(y_val_proba, axis=1)
        if X_test is not None:
            y_test_proba = model.predict_proba(X_test)
            y_test_pred = np.argmax(y_test_proba, axis=1)

        if y_test is not None and y_test_pred is not None and y_test_proba is not None:
            record.testing_metrics = compute_all_classification_metrics(y_test, y_test_pred, y_test_proba)
        if y_val is not None and y_val_pred is not None and y_val_proba is not None:
            record.validation_metrics = compute_all_classification_metrics(y_val, y_val_pred, y_val_proba)

        metadata = manager.load_json("metadata")
        feature_names = metadata["feature_names"]
        training_statistics = TrainingStatistics(**manager.load_json("training_report"))
        feature_importance = model.feature_importance(feature_names)

        report = ClassificationEvaluator().evaluate(
            dataset_summary={
                "n_samples": len(dataset), "target": target_name,
                "symbol": dataset.metadata["symbol"].iloc[0] if len(dataset) else None,
                "timeframe": dataset.metadata["timeframe"].iloc[0] if len(dataset) else None,
            },
            feature_names=feature_names, classes=list(cfg.classes), training_statistics=training_statistics,
            y_val=y_val, y_val_pred=y_val_pred, y_val_proba=y_val_proba,
            y_test=y_test, y_test_pred=y_test_pred, y_test_proba=y_test_proba,
            feature_importance=feature_importance, coefficients=model.coefficients,
            bootstrap_coefficients=model.bootstrap_coefficients,
        )
        manager.save_json(report, "evaluation_report")
        self.experiment_manager.log(record)
