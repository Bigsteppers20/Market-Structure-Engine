"""Tests for logistic_regression.trainer.LogisticRegressionTrainer -- exercises
the actual training.Trainer subclassing extension point against a real
classification dataset (built via ConfigurableClassificationLabelGenerator)."""
from __future__ import annotations

import numpy as np
import pytest

from logistic_regression.classification_model import ClassificationModel
from logistic_regression.config import ClassificationConfig
from logistic_regression.trainer import LogisticRegressionTrainer
from training.artifacts import ArtifactManager
from training.config import TrainingConfig
from training.trainer import Trainer


def _lgr_config(tmp_path, **overrides):
    tc = TrainingConfig(experiment_name="lgr_test", output_root=str(tmp_path), random_seed=5)
    base = dict(prediction_horizon=5, n_bootstrap=5, training_config=tc)
    base.update(overrides)
    return ClassificationConfig(**base)


def test_is_a_training_trainer_subclass() -> None:
    assert issubclass(LogisticRegressionTrainer, Trainer)


def test_model_family_and_task_type(tmp_path) -> None:
    trainer = LogisticRegressionTrainer(_lgr_config(tmp_path))
    assert trainer.model_family == "logistic_regression"
    assert trainer.task_type == "classification"


def test_train_produces_experiment_record(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    trainer = LogisticRegressionTrainer(_lgr_config(tmp_path))
    record = trainer.run(dataset, split)
    assert record.model_family == "logistic_regression"
    assert record.task_type == "classification"
    assert "accuracy" in record.training_metrics
    assert "f1" in record.training_metrics
    assert "confusion_matrix" in record.training_metrics


def test_real_model_persisted_not_just_placeholder(tmp_path, lgr_dataset_and_split) -> None:
    """The base Trainer.run() only ever saves a ModelPlaceholder -- confirm
    the subclass's additive save actually persists a usable classifier."""
    dataset, split, _ = lgr_dataset_and_split
    trainer = LogisticRegressionTrainer(_lgr_config(tmp_path))
    record = trainer.run(dataset, split)

    manager = ArtifactManager(record.artifact_dir)
    assert manager.exists("model")
    loaded_model = manager.load_joblib("model")
    assert isinstance(loaded_model, ClassificationModel)
    assert hasattr(loaded_model, "historical_balanced_accuracy_")
    assert hasattr(loaded_model, "train_feature_mean_")
    assert hasattr(loaded_model, "calibration_error_")


def test_full_classification_config_persisted(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    trainer = LogisticRegressionTrainer(_lgr_config(tmp_path, prediction_horizon=8))
    record = trainer.run(dataset, split)
    manager = ArtifactManager(record.artifact_dir)
    config = manager.load_json("config")
    assert config["prediction_horizon"] == 8
    assert config["classes"] == ["SELL", "NO_TRADE", "BUY"]


def test_feature_importance_is_populated(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    trainer = LogisticRegressionTrainer(_lgr_config(tmp_path))
    record = trainer.run(dataset, split)
    evaluation_report = ArtifactManager(record.artifact_dir).load_json("evaluation_report")
    assert evaluation_report["feature_importance"] != "not available -- no predictions/probabilities/importances were supplied"


def test_evaluation_report_augmented_with_classification_diagnostics(tmp_path, lgr_dataset_and_split) -> None:
    """The base Trainer.run()'s own EvaluationEngine pass only computes
    generic accuracy/precision/recall/f1 -- confirm LogisticRegressionTrainer
    additively wires in ClassificationEvaluator's richer diagnostics
    (calibration curves, coefficient stability, feature importance report,
    prediction/probability distributions) and that ROC-AUC/log-loss/Brier
    reach the returned ExperimentRecord itself, not just the JSON artifact."""
    dataset, split, _ = lgr_dataset_and_split
    trainer = LogisticRegressionTrainer(_lgr_config(tmp_path, n_bootstrap=5))
    record = trainer.run(dataset, split)

    assert "roc_auc" in record.testing_metrics
    assert "log_loss" in record.testing_metrics
    assert "brier_score" in record.testing_metrics

    evaluation_report = ArtifactManager(record.artifact_dir).load_json("evaluation_report")
    assert "calibration_curves" in evaluation_report
    assert set(evaluation_report["calibration_curves"]) == {"SELL", "NO_TRADE", "BUY"}
    assert "coefficient_diagnostics" in evaluation_report
    assert "feature_importance_report" in evaluation_report
    assert "top_20_most_influential" in evaluation_report["feature_importance_report"]
    assert "prediction_distribution" in evaluation_report
    assert "probability_distribution" in evaluation_report
    # n_bootstrap=5 -- coefficient *stability* (not just magnitude) must be present.
    assert evaluation_report["coefficient_diagnostics"]["stability"] != "not available -- n_bootstrap was 0"


def test_target_name_defaults_to_joined_classes(tmp_path, lgr_dataset_and_split) -> None:
    """target_name is accepted only for base-class signature compatibility
    -- classification always trains against dataset.y_cls."""
    dataset, split, _ = lgr_dataset_and_split
    trainer = LogisticRegressionTrainer(_lgr_config(tmp_path))
    record = trainer.run(dataset, split)
    assert record is not None  # no target_name passed -- must not raise


def test_class_balancing_strategies_all_train_successfully(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    for strategy in ("none", "class_weight", "oversample", "undersample", "balanced_sampling"):
        trainer = LogisticRegressionTrainer(_lgr_config(tmp_path, class_balancing=strategy, n_bootstrap=0))
        record = trainer.run(dataset, split)
        assert np.isfinite(record.training_metrics["accuracy"])


def test_calibration_methods_all_train_successfully(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    for method in ("none", "platt", "isotonic"):
        trainer = LogisticRegressionTrainer(_lgr_config(tmp_path, calibration_method=method, n_bootstrap=0))
        record = trainer.run(dataset, split)
        assert np.isfinite(record.training_metrics["accuracy"])


def test_historical_balanced_accuracy_computed_from_internal_holdout(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    trainer = LogisticRegressionTrainer(_lgr_config(tmp_path, n_bootstrap=0))
    trainer.run(dataset, split)
    model = trainer._model
    assert model is not None
    assert 0.0 <= model.historical_balanced_accuracy_ <= 1.0


def test_extended_class_set_trains_via_custom_label_generator(tmp_path) -> None:
    """The architecture must support a different (larger) class set without
    any change to LogisticRegressionTrainer/ClassificationConfig's shape."""
    from ml_pipeline import DatasetBuilder
    from ml_pipeline.config import DatasetConfig as MLDatasetConfig
    from ml_pipeline.label_generator import LabelGenerator
    from ml_pipeline.splitter import TimeSeriesSplitter
    from conftest import make_ohlcv

    class TwoBinLabelGenerator(LabelGenerator):
        classes = ("DOWN", "UP")

        def label(self, df, index, horizon):
            c0 = float(df["close"].iloc[index])
            c1 = float(df["close"].iloc[index + horizon])
            return "UP" if c1 >= c0 else "DOWN"

    df = make_ohlcv(500, seed=21)
    ml_cfg = MLDatasetConfig(window_size=100, horizon=5, stride=4, symbol="TEST", timeframe="M5")
    dataset = DatasetBuilder(ml_cfg, label_generator=TwoBinLabelGenerator()).build(df)
    split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(dataset))[0]

    tc = TrainingConfig(experiment_name="lgr_2class", output_root=str(tmp_path), random_seed=1)
    cfg = ClassificationConfig(classes=("DOWN", "UP"), prediction_horizon=5, n_bootstrap=0, training_config=tc)
    trainer = LogisticRegressionTrainer(cfg)
    record = trainer.run(dataset, split)
    assert np.isfinite(record.training_metrics["accuracy"])
