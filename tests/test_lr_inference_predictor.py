"""Tests for linear_regression.inference and linear_regression.predictor."""
from __future__ import annotations

import numpy as np
import pytest

from linear_regression.config import RegressionConfig
from linear_regression.exceptions import ModelNotTrainedError, VersionMismatchError
from linear_regression.inference import RegressionInferencePipeline
from linear_regression.predictor import RegressionPredictor
from linear_regression.trainer import LinearRegressionTrainer
from training.config import TrainingConfig


def _train_one(tmp_path, dataset, split, target, **overrides):
    tc = TrainingConfig(experiment_name="lr_infer_test", output_root=str(tmp_path), random_seed=3)
    base = dict(targets=[target], prediction_horizon=5, model_type="linear", n_bootstrap=6, training_config=tc)
    base.update(overrides)
    cfg = RegressionConfig(**base)
    trainer = LinearRegressionTrainer(cfg)
    record = trainer.run(dataset, split, target_name=target)
    return record


def test_inference_pipeline_load_and_predict(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, _ = lr_dataset_and_split
    record = _train_one(tmp_path, dataset, split, "next_close")

    pipeline = RegressionInferencePipeline(record.artifact_dir, feature_version="1.0.0").load()
    assert pipeline.model is not None
    assert pipeline.model_version.regression_target == "next_close"
    assert pipeline.model_version.prediction_horizon == 5

    X = dataset.X[split.test_idx][:1]
    point, std = pipeline.predict(X, dataset.feature_names)
    assert np.isfinite(point).all()
    assert std is not None


def test_predict_before_load_raises(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, _ = lr_dataset_and_split
    record = _train_one(tmp_path, dataset, split, "next_close")
    pipeline = RegressionInferencePipeline(record.artifact_dir)
    with pytest.raises(ModelNotTrainedError):
        pipeline.predict(dataset.X[:1], dataset.feature_names)


def test_strict_version_mismatch_raises(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, _ = lr_dataset_and_split
    record = _train_one(tmp_path, dataset, split, "next_close")
    pipeline = RegressionInferencePipeline(record.artifact_dir, feature_version="999.0.0", strict=True)
    with pytest.raises(VersionMismatchError):
        pipeline.load()


def test_non_strict_version_mismatch_collects_warning(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, _ = lr_dataset_and_split
    record = _train_one(tmp_path, dataset, split, "next_close")
    pipeline = RegressionInferencePipeline(record.artifact_dir, feature_version="999.0.0", strict=False)
    pipeline.load()  # must not raise
    assert pipeline.version_warnings


def test_predictor_composes_multiple_targets(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, _ = lr_dataset_and_split
    targets = ["next_close", "next_high", "expected_pip_movement"]
    pipelines = {}
    for target in targets:
        record = _train_one(tmp_path, dataset, split, target)
        pipelines[target] = RegressionInferencePipeline(record.artifact_dir).load()

    predictor = RegressionPredictor(pipelines, pip_size=0.0001)
    X = dataset.X[split.test_idx][:1]
    names = dataset.feature_names

    # Use MarketState-free path via internal helper: build a raw prediction
    # using the same array both feature_mapper.extract_feature_vector would
    # produce, to test predictor composition without needing a live MarketState.
    from linear_regression.feature_mapper import extract_feature_vector

    class _FakeMarketState:
        def to_vector(self):
            return X.ravel(), names
        def to_dict(self):
            return {f"{n}_valid": 1.0 for n in names[:5]}  # minimal, some _valid keys

    prediction = predictor.predict(_FakeMarketState(), symbol="EUR_USD", timeframe="M5")
    assert prediction.expected_close is not None
    assert prediction.expected_high is not None
    assert prediction.expected_pip_move is not None
    assert prediction.expected_low is None  # not trained -- must stay None
    assert 0.0 <= prediction.prediction_confidence <= 100.0
    assert prediction.symbol == "EUR_USD"
    assert prediction.prediction_horizon == 5
    assert len(prediction.explanation) >= 2
    assert set(prediction.confidence_breakdown) == set(targets)


def test_predictor_requires_at_least_one_pipeline() -> None:
    with pytest.raises(ValueError):
        RegressionPredictor({})
