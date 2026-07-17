"""Model-agnostic training orchestration.

:class:`Trainer` is abstract -- it implements every *infrastructure*
responsibility (load data, preprocess, scale, select features, compute
metrics, build the evaluation report, persist artifacts, log the
experiment, register the model) and delegates every *modeling* decision to
four small abstract hooks. A concrete subclass added later (e.g. for
scikit-learn's ``LinearRegression``, XGBoost, a PyTorch network, ...)
implements only those hooks; nothing here changes per model family.

No hook implementation lives in this module -- see the module docstring
note in ``tests/test_training_trainer.py`` for the minimal test-only stub
used to exercise this orchestration logic.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from ml_pipeline import Dataset, FeaturePipeline, FeatureScaler, FeatureSelector
from ml_pipeline.splitter import SplitResult

from .artifacts import ArtifactManager, ModelPlaceholder
from .config import TrainingConfig
from .evaluator import EvaluationEngine
from .experiment import ExperimentManager, ExperimentRecord
from .metrics import TrainingStatistics, compute_classification_metrics, compute_regression_metrics
from .registry import ModelMetadata, ModelRegistry
from .utils import Timer, ensure_dir, new_id, set_random_seed, utc_timestamp
from .versioning import FeatureSchema, assert_schema_compatible, current_version_info


class Trainer(ABC):
    """Abstract, model-agnostic training orchestrator. See module docstring."""

    def __init__(self, config: TrainingConfig, repo_dir: Optional[Path] = None) -> None:
        self.config = config
        self.repo_dir = repo_dir
        self.experiment_manager = ExperimentManager(config.output_dir)
        self.model_registry = ModelRegistry(config.output_dir)
        self._fitted_pipeline: Optional[FeaturePipeline] = None
        self._fitted_scaler: Optional[FeatureScaler] = None
        self._fitted_selector: Optional[FeatureSelector] = None

    # ------------------------------------------------------------------ #
    # Abstract hooks -- implemented by a concrete, model-specific subclass.
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def model_family(self) -> str:
        """Short identifier, e.g. ``"linear_regression"``, ``"xgboost"``."""

    @property
    @abstractmethod
    def task_type(self) -> str:
        """``"regression"`` or ``"classification"``."""

    @abstractmethod
    def build_model(self, hyperparameters: Dict[str, Any]) -> Any:
        """Construct an untrained model object from hyperparameters."""

    @abstractmethod
    def fit_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Fit ``model`` on training data; return the fitted model."""

    @abstractmethod
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Produce predictions for ``X``."""

    # Optional hooks -- safe no-op defaults; override only if applicable.
    def predict_proba(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        return None

    def feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        return None

    # ------------------------------------------------------------------ #
    # "Load datasets" responsibility
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_dataset(path: str | Path) -> Dataset:
        """Load a ``ml_pipeline.Dataset`` previously written by ``DatasetExporter``."""
        path = Path(path)
        if path.suffix == ".joblib":
            return joblib.load(path)
        if path.suffix in (".pkl", ".pickle"):
            with open(path, "rb") as fh:
                return pickle.load(fh)
        raise ValueError(
            f"Unsupported dataset file extension: {path.suffix!r}. "
            "Use a .joblib or .pkl file written by ml_pipeline.DatasetExporter."
        )

    # ------------------------------------------------------------------ #
    def run(
        self,
        dataset: Dataset,
        split: SplitResult,
        *,
        target_name: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        expected_schema: Optional[FeatureSchema] = None,
    ) -> ExperimentRecord:
        """Run the full train/evaluate/persist/log/register pipeline once.

        Parameters
        ----------
        dataset:
            A ``ml_pipeline.Dataset`` (in-memory or loaded via
            :meth:`load_dataset`).
        split:
            A chronological ``ml_pipeline.splitter.SplitResult`` -- splitting
            itself is the Dataset Builder's responsibility, not the
            Trainer's.
        target_name:
            Key into ``dataset.y_reg`` (regression) -- ignored for
            classification, which always uses ``dataset.y_cls``.
        expected_schema:
            If provided, the dataset's feature schema must exactly match it
            (see :mod:`training.versioning`) or this raises
            ``SchemaMismatchError`` before any training happens.
        """
        set_random_seed(self.config.random_seed)
        hyperparameters = dict(hyperparameters or {})
        experiment_id = new_id("exp")
        started_at = utc_timestamp()

        version_info = current_version_info(self.config.feature_version)
        schema = FeatureSchema.from_feature_names(dataset.feature_names, version_info)
        if expected_schema is not None:
            assert_schema_compatible(expected_schema, dataset.feature_names, dataset.X)

        y = self._select_target(dataset, target_name)
        y_train, y_val, y_test = y[split.train_idx], y[split.val_idx], y[split.test_idx]

        with Timer() as timer:
            X_train, X_val, X_test, feature_names = self._preprocess(dataset, split, y_train)

            model = self.build_model(hyperparameters)
            model = self.fit_model(model, X_train, y_train)

            train_pred = self.predict(model, X_train)
            val_pred = self.predict(model, X_val) if X_val.shape[0] else None
            test_pred = self.predict(model, X_test) if X_test.shape[0] else None

        finished_at = utc_timestamp()
        compute_fn = (
            compute_regression_metrics if self.task_type == "regression" else compute_classification_metrics
        )
        training_metrics = compute_fn(y_train, train_pred)
        validation_metrics = compute_fn(y_val, val_pred) if val_pred is not None else {}
        testing_metrics = compute_fn(y_test, test_pred) if test_pred is not None else {}

        training_stats = TrainingStatistics(
            duration_seconds=timer.elapsed_seconds,
            n_train_samples=len(split.train_idx), n_val_samples=len(split.val_idx),
            n_test_samples=len(split.test_idx), n_features=len(feature_names),
            random_seed=self.config.random_seed, started_at=started_at, finished_at=finished_at,
        )

        test_proba = self.predict_proba(model, X_test) if X_test.shape[0] else None
        importance = self.feature_importance(model, feature_names)

        evaluation_report = EvaluationEngine().evaluate(
            task_type=self.task_type,
            dataset_summary={
                "n_samples": len(dataset), "target": target_name,
                "symbol": dataset.metadata["symbol"].iloc[0] if len(dataset) else None,
                "timeframe": dataset.metadata["timeframe"].iloc[0] if len(dataset) else None,
            },
            feature_names=feature_names, training_statistics=training_stats,
            y_val=y_val, y_val_pred=val_pred, y_test=y_test, y_test_pred=test_pred,
            y_test_proba=test_proba, feature_importance=importance,
        )

        artifact_dir = ensure_dir(Path(self.config.output_dir) / "artifacts" / experiment_id)
        artifact_manager = ArtifactManager(artifact_dir)
        artifact_manager.save_bundle(
            scaler=self._fitted_scaler,
            feature_selector=self._fitted_selector,
            feature_pipeline=self._fitted_pipeline,
            config=self.config.to_dict(),
            metadata={
                "experiment_id": experiment_id, "model_family": self.model_family,
                "task_type": self.task_type, "target_name": target_name,
                "feature_names": feature_names,
            },
            training_report=training_stats.to_dict(),
            evaluation_report=evaluation_report.to_dict(),
            feature_schema=schema.to_dict(),
            model_placeholder=ModelPlaceholder(self.model_family, self.task_type),
        )

        record = self.experiment_manager.new_record(
            config=self.config, model_family=self.model_family, task_type=self.task_type,
            hyperparameters=hyperparameters, training_metrics=training_metrics,
            validation_metrics=validation_metrics, testing_metrics=testing_metrics,
            training_duration_seconds=timer.elapsed_seconds, artifact_dir=artifact_dir,
            repo_dir=self.repo_dir, experiment_id=experiment_id,
        )
        self.experiment_manager.log(record)

        self.model_registry.register(ModelMetadata(
            model_name=self.config.experiment_name, version=self.config.strategy_version,
            training_date=finished_at, feature_count=len(feature_names),
            feature_schema_version=schema.version_info.schema_version,
            training_dataset_version=self.config.dataset_version,
            performance_metrics=testing_metrics or validation_metrics or training_metrics,
            supported_timeframes=self.config.supported_timeframes,
            supported_symbols=self.config.supported_symbols,
            training_strategy=self.config.training_strategy,
            artifact_dir=str(artifact_dir), model_family=self.model_family,
            task_type=self.task_type, experiment_id=experiment_id,
        ))

        return record

    # ------------------------------------------------------------------ #
    def _select_target(self, dataset: Dataset, target_name: str) -> np.ndarray:
        if self.task_type == "regression":
            if target_name not in dataset.y_reg:
                raise KeyError(
                    f"Regression target {target_name!r} not in dataset.y_reg: {list(dataset.y_reg)}"
                )
            return dataset.y_reg[target_name]
        if self.task_type == "classification":
            if dataset.y_cls is None:
                raise ValueError("Dataset has no classification labels (y_cls is None).")
            return dataset.y_cls
        raise ValueError(f"Unknown task_type {self.task_type!r} (must be 'regression' or 'classification').")

    def _preprocess(
        self, dataset: Dataset, split: SplitResult, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Fit-on-train-only feature pipeline -> scaler -> optional selector."""
        pipeline_cfg = self.config.feature_pipeline_config
        fp = FeaturePipeline(pipeline_cfg)
        x_train, names = fp.fit_transform(dataset.X[split.train_idx], dataset.feature_names)
        x_val, _ = self._transform_or_empty(fp, dataset, split.val_idx, names)
        x_test, _ = self._transform_or_empty(fp, dataset, split.test_idx, names)
        self._fitted_pipeline = fp

        scaler = FeatureScaler(self.config.scaler)
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val) if x_val.shape[0] else x_val
        x_test = scaler.transform(x_test) if x_test.shape[0] else x_test
        self._fitted_scaler = scaler

        self._fitted_selector = None
        if self.config.feature_selector:
            selector = FeatureSelector(
                self.config.feature_selector, target_type=self.task_type,
                **self.config.feature_selector_kwargs,
            )
            x_train, sel_names = selector.fit_transform(x_train, y_train, names)
            x_val, _ = (selector.transform(x_val, names) if x_val.shape[0] else (x_val, sel_names))
            x_test, _ = (selector.transform(x_test, names) if x_test.shape[0] else (x_test, sel_names))
            names = sel_names
            self._fitted_selector = selector

        return x_train, x_val, x_test, names

    @staticmethod
    def _transform_or_empty(
        fp: FeaturePipeline, dataset: Dataset, idx: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        if idx.shape[0] == 0:
            return np.empty((0, len(names))), names
        return fp.transform(dataset.X[idx], dataset.feature_names)
