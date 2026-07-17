"""Tests for training.inference -- artifact loading, version enforcement,
and feature-vector preparation (no prediction logic exists to test)."""
from __future__ import annotations

import pytest

from conftest import make_ohlcv
from ml_pipeline import DatasetBuilder
from ml_pipeline.config import DatasetConfig as MLDatasetConfig
from ml_pipeline.splitter import TimeSeriesSplitter
from training.config import TrainingConfig
from training.inference import InferencePipeline
from training.versioning import VersionMismatchError

from _training_stub import MeanBaselineStub


@pytest.fixture(scope="module")
def trained_artifact_dir(tmp_path_factory):
    df = make_ohlcv(300, seed=31)
    ml_cfg = MLDatasetConfig(window_size=80, horizon=2, stride=5, symbol="TEST", timeframe="M5")
    dataset = DatasetBuilder(ml_cfg).build(df)
    split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(dataset))[0]

    out_root = tmp_path_factory.mktemp("training_output")
    cfg = TrainingConfig(experiment_name="inference_test", output_root=str(out_root))
    trainer = MeanBaselineStub(cfg)
    record = trainer.run(dataset, split, target_name="next_return")
    return record.artifact_dir, dataset, split


def test_load_succeeds_and_populates_schema(trained_artifact_dir) -> None:
    artifact_dir, dataset, _ = trained_artifact_dir
    pipeline = InferencePipeline(artifact_dir).load()
    assert pipeline.feature_schema is not None
    assert pipeline.feature_schema.feature_count == dataset.X.shape[1]
    assert pipeline.scaler is not None


def test_prepare_before_load_raises(trained_artifact_dir) -> None:
    artifact_dir, _, _ = trained_artifact_dir
    pipeline = InferencePipeline(artifact_dir)
    with pytest.raises(RuntimeError):
        pipeline.prepare(None, [])


def test_prepare_returns_finite_processed_matrix(trained_artifact_dir) -> None:
    artifact_dir, dataset, split = trained_artifact_dir
    pipeline = InferencePipeline(artifact_dir).load()
    X_new = dataset.X[split.test_idx][:3]
    X_prepared, names = pipeline.prepare(X_new, dataset.feature_names)
    assert X_prepared.shape[0] == 3
    assert len(names) == X_prepared.shape[1]
    import numpy as np
    assert np.isfinite(X_prepared).all()


def test_prepare_rejects_wrong_feature_count(trained_artifact_dir) -> None:
    artifact_dir, dataset, split = trained_artifact_dir
    pipeline = InferencePipeline(artifact_dir).load()
    X_new = dataset.X[split.test_idx][:3, :5]  # truncated -- wrong count
    with pytest.raises(Exception):  # SchemaMismatchError (a VersionMismatchError)
        pipeline.prepare(X_new, dataset.feature_names[:5])


def test_prepare_rejects_wrong_feature_order(trained_artifact_dir) -> None:
    artifact_dir, dataset, split = trained_artifact_dir
    pipeline = InferencePipeline(artifact_dir).load()
    X_new = dataset.X[split.test_idx][:3]
    shuffled_names = list(reversed(dataset.feature_names))
    X_shuffled = X_new[:, ::-1]
    with pytest.raises(Exception):
        pipeline.prepare(X_shuffled, shuffled_names)


def test_strict_mode_raises_on_feature_version_mismatch(trained_artifact_dir) -> None:
    artifact_dir, _, _ = trained_artifact_dir
    pipeline = InferencePipeline(artifact_dir, feature_version="99.0.0", strict=True)
    with pytest.raises(VersionMismatchError):
        pipeline.load()


def test_non_strict_mode_collects_warnings_instead_of_raising(trained_artifact_dir) -> None:
    artifact_dir, _, _ = trained_artifact_dir
    pipeline = InferencePipeline(artifact_dir, feature_version="99.0.0", strict=False)
    pipeline.load()  # must not raise
    assert pipeline.version_warnings  # but the mismatch is recorded


def test_missing_feature_schema_raises_file_not_found(tmp_path) -> None:
    pipeline = InferencePipeline(tmp_path)  # empty directory, nothing trained here
    with pytest.raises(FileNotFoundError):
        pipeline.load()


def test_no_selector_means_feature_selector_stays_none(trained_artifact_dir) -> None:
    artifact_dir, _, _ = trained_artifact_dir
    pipeline = InferencePipeline(artifact_dir).load()
    assert pipeline.feature_selector is None  # this run used no selector
