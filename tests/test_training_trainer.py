"""Tests for training.trainer -- the abstract orchestrator, exercised via
the test-only stub trainers in _training_stub.py (see that module's
docstring: they are not ML algorithms, just test doubles)."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import make_ohlcv
from ml_pipeline import DatasetBuilder
from ml_pipeline.config import DatasetConfig as MLDatasetConfig
from ml_pipeline.splitter import TimeSeriesSplitter
from training.config import TrainingConfig
from training.versioning import FeatureSchema, VersionMismatchError, current_version_info

from _training_stub import MajorityClassStub, MeanBaselineStub


@pytest.fixture(scope="module")
def small_dataset():
    df = make_ohlcv(400, seed=21)
    cfg = MLDatasetConfig(window_size=100, horizon=2, stride=5, symbol="TEST", timeframe="M5")
    return DatasetBuilder(cfg).build(df)


@pytest.fixture()
def split(small_dataset):
    return TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(small_dataset))[0]


def test_regression_run_produces_complete_record(tmp_path, small_dataset, split) -> None:
    cfg = TrainingConfig(experiment_name="mean_baseline", output_root=str(tmp_path), random_seed=5)
    trainer = MeanBaselineStub(cfg)
    record = trainer.run(small_dataset, split, target_name="next_return")

    assert record.model_family == "mean_baseline_stub"
    assert record.task_type == "regression"
    assert set(record.training_metrics) >= {"mae", "mse", "rmse", "r2", "mape"}
    assert record.testing_metrics  # non-empty (test split is non-trivial)
    assert record.training_duration_seconds >= 0.0
    assert record.random_seed == 5


def test_regression_run_writes_all_artifacts(tmp_path, small_dataset, split) -> None:
    cfg = TrainingConfig(experiment_name="mean_baseline2", output_root=str(tmp_path))
    trainer = MeanBaselineStub(cfg)
    record = trainer.run(small_dataset, split, target_name="next_close")

    from training.artifacts import ArtifactManager
    mgr = ArtifactManager(record.artifact_dir)
    for name in ("scaler", "config", "metadata", "training_report",
                 "evaluation_report", "feature_schema", "model_placeholder"):
        assert mgr.exists(name), f"missing artifact: {name}"
    assert not mgr.exists("model")  # stub never provides a real model
    assert not mgr.exists("feature_selector")  # none configured


def test_run_registers_model_in_registry(tmp_path, small_dataset, split) -> None:
    cfg = TrainingConfig(experiment_name="registry_test", strategy_version="7.0.0",
                          output_root=str(tmp_path), supported_symbols=["EUR_USD"])
    trainer = MeanBaselineStub(cfg)
    trainer.run(small_dataset, split, target_name="next_return")

    meta = trainer.model_registry.get("registry_test", version="7.0.0")
    assert meta.feature_count == small_dataset.X.shape[1]
    assert meta.supported_symbols == ["EUR_USD"]
    assert meta.model_family == "mean_baseline_stub"


def test_run_logs_experiment_retrievable(tmp_path, small_dataset, split) -> None:
    cfg = TrainingConfig(experiment_name="log_test", output_root=str(tmp_path))
    trainer = MeanBaselineStub(cfg)
    record = trainer.run(small_dataset, split, target_name="next_return")
    reloaded = trainer.experiment_manager.load(record.experiment_id)
    assert reloaded == record


def test_classification_run_with_feature_selector(tmp_path, small_dataset, split) -> None:
    cfg = TrainingConfig(
        experiment_name="cls_test", output_root=str(tmp_path), scaler="minmax",
        feature_selector="variance", feature_selector_kwargs={"variance_threshold": 1e-8},
    )
    trainer = MajorityClassStub(cfg)
    record = trainer.run(small_dataset, split, target_name="")
    assert record.task_type == "classification"
    assert "accuracy" in record.training_metrics
    assert "confusion_matrix" in record.training_metrics

    from training.artifacts import ArtifactManager
    mgr = ArtifactManager(record.artifact_dir)
    assert mgr.exists("feature_selector")
    schema = mgr.load_json("feature_schema")
    # feature_schema always reflects the RAW engine output, not post-selection
    assert schema["feature_count"] == small_dataset.X.shape[1]


def test_unknown_regression_target_raises_keyerror(tmp_path, small_dataset, split) -> None:
    cfg = TrainingConfig(output_root=str(tmp_path))
    trainer = MeanBaselineStub(cfg)
    with pytest.raises(KeyError):
        trainer.run(small_dataset, split, target_name="not_a_real_target")


def test_expected_schema_mismatch_blocks_training(tmp_path, small_dataset, split) -> None:
    cfg = TrainingConfig(output_root=str(tmp_path))
    trainer = MeanBaselineStub(cfg)
    bad_schema = FeatureSchema.from_feature_names(["only_one"], current_version_info())
    with pytest.raises(VersionMismatchError):
        trainer.run(small_dataset, split, target_name="next_return", expected_schema=bad_schema)


def test_load_dataset_rejects_unsupported_extension(tmp_path) -> None:
    cfg = TrainingConfig(output_root=str(tmp_path))
    trainer = MeanBaselineStub(cfg)
    bogus = tmp_path / "data.csv"
    bogus.write_text("not a dataset")
    with pytest.raises(ValueError):
        trainer.load_dataset(bogus)


def test_trainer_is_abstract() -> None:
    from training.trainer import Trainer
    with pytest.raises(TypeError):
        Trainer(TrainingConfig())  # cannot instantiate the abstract base directly
