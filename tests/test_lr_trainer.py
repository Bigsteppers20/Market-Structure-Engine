"""Tests for linear_regression.trainer.LinearRegressionTrainer -- exercises
the actual training.Trainer subclassing extension point."""
from __future__ import annotations

import numpy as np
import pytest

from linear_regression.config import RegressionConfig
from linear_regression.trainer import LinearRegressionTrainer
from training.artifacts import ArtifactManager
from training.config import TrainingConfig


def _lr_config(tmp_path, **overrides):
    tc = TrainingConfig(experiment_name="lr_test", output_root=str(tmp_path), random_seed=5)
    base = dict(targets=["next_close"], prediction_horizon=5, model_type="linear",
                n_bootstrap=5, training_config=tc)
    base.update(overrides)
    return RegressionConfig(**base)


def test_is_a_training_trainer_subclass() -> None:
    from training.trainer import Trainer
    assert issubclass(LinearRegressionTrainer, Trainer)


def test_train_produces_experiment_record(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, _ = lr_dataset_and_split
    trainer = LinearRegressionTrainer(_lr_config(tmp_path))
    record = trainer.run(dataset, split, target_name="next_close")
    assert record.model_family == "linear_regression:linear"
    assert record.task_type == "regression"
    assert "mae" in record.training_metrics
    assert "explained_variance" in record.training_metrics  # confirms metrics.py's extension is live


def test_real_model_persisted_not_just_placeholder(tmp_path, lr_dataset_and_split) -> None:
    """The base Trainer.run() only ever saves a ModelPlaceholder -- confirm
    the subclass's additive save actually persists a usable model."""
    dataset, split, _ = lr_dataset_and_split
    trainer = LinearRegressionTrainer(_lr_config(tmp_path))
    record = trainer.run(dataset, split, target_name="next_close")

    manager = ArtifactManager(record.artifact_dir)
    assert manager.exists("model")
    loaded_model = manager.load_joblib("model")
    from linear_regression.regression_model import RegressionModel
    assert isinstance(loaded_model, RegressionModel)
    assert hasattr(loaded_model, "residual_std_")
    assert hasattr(loaded_model, "historical_r2_")
    assert hasattr(loaded_model, "train_feature_mean_")


def test_full_regression_config_persisted(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, _ = lr_dataset_and_split
    trainer = LinearRegressionTrainer(_lr_config(tmp_path, prediction_horizon=10))
    record = trainer.run(dataset, split, target_name="next_close")
    manager = ArtifactManager(record.artifact_dir)
    config = manager.load_json("config")
    assert config["prediction_horizon"] == 10
    assert config["model_type"] == "linear"


def test_feature_importance_is_populated(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, _ = lr_dataset_and_split
    trainer = LinearRegressionTrainer(_lr_config(tmp_path))
    record = trainer.run(dataset, split, target_name="next_close")
    evaluation_report = ArtifactManager(record.artifact_dir).load_json("evaluation_report")
    assert evaluation_report["feature_importance"] != "not available -- no predictions/probabilities/importances were supplied"


def test_multiple_model_types_train_successfully(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, _ = lr_dataset_and_split
    for model_type in ("linear", "ridge", "lasso", "elasticnet"):
        trainer = LinearRegressionTrainer(_lr_config(tmp_path, model_type=model_type, n_bootstrap=0))
        record = trainer.run(dataset, split, target_name="next_close")
        assert np.isfinite(record.training_metrics["mae"])


def test_different_targets_produce_different_models(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, _ = lr_dataset_and_split
    t1 = LinearRegressionTrainer(_lr_config(tmp_path, n_bootstrap=0))
    r1 = t1.run(dataset, split, target_name="next_close")
    t2 = LinearRegressionTrainer(_lr_config(tmp_path, n_bootstrap=0))
    r2 = t2.run(dataset, split, target_name="expected_pip_movement")
    assert r1.training_metrics["mae"] != r2.training_metrics["mae"]
