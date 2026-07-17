"""``LinearRegressionTrainer`` -- a concrete subclass of ``training.Trainer``.

This is the extension point ``training/TRAINING_INFRASTRUCTURE_REPORT.md``
was written for: implement the four abstract hooks
(``model_family``, ``task_type``, ``build_model``, ``fit_model``,
``predict``) plus the optional ``feature_importance`` hook, and every other
responsibility -- dataset loading, ``FeaturePipeline``/``FeatureScaler``/
``FeatureSelector`` orchestration, metric computation, evaluation-report
generation, experiment logging, and model-registry registration -- comes
free from the unmodified base class via ``Trainer.run()``.

Two things ``training.Trainer.run()`` deliberately does *not* do (since the
base infrastructure implements no model): persist the real fitted model
(it only ever saves a ``ModelPlaceholder``), and compute regression-specific,
confidence-supporting statistics. Both are added here as pure *extensions*
around an unmodified call to ``super().run()`` -- see ``run()`` below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import r2_score
from training.artifacts import ArtifactManager
from training.experiment import ExperimentRecord
from training.trainer import Trainer
from ml_pipeline.splitter import SplitResult
from ml_pipeline.dataset_builder import Dataset

from .config import RegressionConfig
from .cross_validation import cross_validate
from .model_scoring import compute_model_health_scores
from .regression_model import RegressionModel


class LinearRegressionTrainer(Trainer):
    """Trains one regression target using scikit-learn linear estimators.

    One instance trains exactly one target (``training.Trainer.run()``'s
    contract is single-target); ``regression_engine.py`` composes several
    instances -- one per configured target -- for multi-output support
    without touching this class or its parent.
    """

    #: Fraction of the (already time-ordered) training split carved out as
    #: an internal, self-contained holdout purely to estimate historical
    #: out-of-sample accuracy for the confidence engine. Never used to fit
    #: the deployed model, and never overlaps the real validation/test
    #: splits that `training.Trainer.run()` computes independently.
    INTERNAL_HOLDOUT_FRACTION = 0.15

    def __init__(self, regression_config: RegressionConfig, repo_dir: Optional[Path] = None) -> None:
        super().__init__(regression_config.training_config, repo_dir=repo_dir)
        self.regression_config = regression_config
        self._model: Optional[RegressionModel] = None
        self._current_target_name: str = regression_config.targets[0]

    # ------------------------------------------------------------------ #
    @property
    def model_family(self) -> str:
        return f"linear_regression:{self.regression_config.model_type}"

    @property
    def task_type(self) -> str:
        return "regression"

    def build_model(self, hyperparameters: Dict[str, Any]) -> RegressionModel:
        return RegressionModel(
            model_type=self.regression_config.model_type,
            n_bootstrap=self.regression_config.n_bootstrap,
            random_state=self.config.random_seed,
            **hyperparameters,
        )

    def fit_model(self, model: RegressionModel, X_train: np.ndarray, y_train: np.ndarray) -> RegressionModel:
        y_train = np.asarray(y_train, dtype=float)

        # Internal, self-contained holdout for a genuine out-of-sample
        # accuracy estimate (see INTERNAL_HOLDOUT_FRACTION docstring).
        n = X_train.shape[0]
        val_size = max(1, int(n * self.INTERNAL_HOLDOUT_FRACTION))
        historical_r2 = 0.0
        historical_rmse = 0.0
        if n - val_size >= 5:
            X_fit, X_holdout = X_train[:-val_size], X_train[-val_size:]
            y_fit, y_holdout = y_train[:-val_size], y_train[-val_size:]
            probe = RegressionModel(self.regression_config.model_type, n_bootstrap=0,
                                     random_state=self.config.random_seed,
                                     **self.regression_config.model_hyperparameters)
            probe.fit(X_fit, y_fit, [self._current_target_name])
            holdout_pred = probe.predict(X_holdout).ravel()
            historical_r2 = float(r2_score(y_holdout, holdout_pred))
            historical_rmse = float(np.sqrt(np.mean((y_holdout - holdout_pred) ** 2)))

        model.fit(X_train, y_train, [self._current_target_name])
        residuals = y_train - model.predict(X_train).ravel()

        # Confidence-supporting stats travel with the model artifact itself
        # (see run() below for why this, rather than the metadata.json
        # artifact, is where they're persisted).
        model.target_std_ = float(np.std(y_train))
        model.residual_std_ = float(np.std(residuals))
        model.historical_r2_ = historical_r2
        model.historical_rmse_ = historical_rmse or float(np.std(residuals))
        model.train_feature_mean_ = X_train.mean(axis=0)
        train_std = X_train.std(axis=0)
        model.train_feature_std_ = np.where(train_std < 1e-9, 1.0, train_std)

        # Phase 8 (Decision Engine metadata) support -- train_r2_ is cheap
        # (the in-sample prediction above is already computed for
        # residuals); cross-validation stays strictly opt-in
        # (RegressionConfig.enable_cross_validation, default False) so
        # existing training runs keep their exact current cost.
        model.train_r2_ = float(r2_score(y_train, model.predict(X_train).ravel()))
        model.cv_mean_r2_ = None
        model.cv_std_r2_ = None
        if self.regression_config.enable_cross_validation and n >= 50:
            try:
                cv_result = cross_validate(
                    X_train, y_train, target_name=self._current_target_name,
                    model_type=self.regression_config.model_type, method="walk_forward_expanding",
                    n_folds=self.regression_config.cv_n_folds, random_state=self.config.random_seed,
                )
                model.cv_mean_r2_ = cv_result.mean_r2
                model.cv_std_r2_ = cv_result.std_r2
            except ValueError:
                pass  # not enough data for the requested fold count -- leave as None, not a training failure

        model.health_scores_ = compute_model_health_scores(
            train_r2=model.train_r2_, holdout_r2=historical_r2, X_train=X_train, y_train=y_train,
            cv_mean_r2=model.cv_mean_r2_, cv_std_r2=model.cv_std_r2_,
        )

        self._model = model
        return model

    def predict(self, model: RegressionModel, X: np.ndarray) -> np.ndarray:
        return model.predict(X).ravel()

    def feature_importance(self, model: RegressionModel, feature_names: List[str]) -> Optional[Dict[str, float]]:
        return model.feature_importance(feature_names)

    # ------------------------------------------------------------------ #
    def run(
        self, dataset: Dataset, split: SplitResult, *, target_name: str,
        hyperparameters: Optional[Dict[str, Any]] = None, expected_schema=None,
    ) -> ExperimentRecord:
        """Identical to ``Trainer.run()`` (called via ``super()``, unmodified),
        plus two additive steps: persist the real fitted model (the base
        class only ever saves a ``ModelPlaceholder`` since it implements no
        algorithm), and persist the *full* ``RegressionConfig`` (the base
        class's ``config.json`` only captures the embedded
        ``training.TrainingConfig``, missing ``prediction_horizon``/
        ``model_type``/``targets`` -- needed at inference time). Both use
        the *same* ``ArtifactManager`` the base class already uses
        internally, targeting the artifact directory it already created."""
        self._current_target_name = target_name
        hyperparameters = hyperparameters or self.regression_config.model_hyperparameters
        record = super().run(
            dataset, split, target_name=target_name, hyperparameters=hyperparameters,
            expected_schema=expected_schema,
        )
        assert self._model is not None
        manager = ArtifactManager(record.artifact_dir)
        manager.save_joblib(self._model, "model")
        manager.save_json(self.regression_config.to_dict(), "config")
        return record
