"""Tests for logistic_regression.inference, logistic_regression.predictor,
and logistic_regression.validator.

Covers: version validation, feature schema validation, class/horizon
mismatch detection, probability-sum validation, and the structured
explanation output -- all against a real trained model, not a mock.
"""
from __future__ import annotations

import numpy as np
import pytest

from logistic_regression.config import ClassificationConfig
from logistic_regression.exceptions import ClassMismatchError, ModelNotTrainedError, VersionMismatchError
from logistic_regression.inference import ClassificationInferencePipeline
from logistic_regression.predictor import ClassificationPredictor
from logistic_regression.probability_engine import assert_probabilities_sum_to_one
from logistic_regression.trainer import LogisticRegressionTrainer
from logistic_regression.validator import validate_classes_and_horizon, validate_for_inference
from logistic_regression.version import current_classification_version
from training.config import TrainingConfig
from training.versioning import FeatureSchema, current_version_info


def _train_one(tmp_path, dataset, split, **overrides):
    tc = TrainingConfig(experiment_name="lgr_infer_test", output_root=str(tmp_path), random_seed=3)
    base = dict(prediction_horizon=5, n_bootstrap=6, training_config=tc)
    base.update(overrides)
    cfg = ClassificationConfig(**base)
    trainer = LogisticRegressionTrainer(cfg)
    record = trainer.run(dataset, split)
    return record


# --------------------------------------------------------------------------- #
# ClassificationInferencePipeline
# --------------------------------------------------------------------------- #
def test_inference_pipeline_load_and_predict(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    record = _train_one(tmp_path, dataset, split)

    pipeline = ClassificationInferencePipeline(record.artifact_dir, feature_version="1.0.0").load()
    assert pipeline.model is not None
    assert pipeline.model_version.classes == ("SELL", "NO_TRADE", "BUY")
    assert pipeline.model_version.prediction_horizon == 5

    X = dataset.X[split.test_idx][:1]
    proba, agreement = pipeline.predict_proba(X, dataset.feature_names)
    assert proba.shape == (1, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert agreement is not None


def test_predict_before_load_raises(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    record = _train_one(tmp_path, dataset, split)
    pipeline = ClassificationInferencePipeline(record.artifact_dir)
    with pytest.raises(ModelNotTrainedError):
        pipeline.predict_proba(dataset.X[:1], dataset.feature_names)


def test_load_without_trained_model_raises(tmp_path) -> None:
    from training.artifacts import ArtifactManager
    from training.versioning import FeatureSchema, current_version_info

    artifact_dir = tmp_path / "empty_exp"
    artifact_dir.mkdir()
    schema = FeatureSchema.from_feature_names(["f0", "f1"], current_version_info("1.0.0"))
    ArtifactManager(artifact_dir).save_json(schema.to_dict(), "feature_schema")
    pipeline = ClassificationInferencePipeline(artifact_dir)
    with pytest.raises(ModelNotTrainedError):
        pipeline.load()


def test_strict_version_mismatch_raises(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    record = _train_one(tmp_path, dataset, split)
    pipeline = ClassificationInferencePipeline(record.artifact_dir, feature_version="999.0.0", strict=True)
    with pytest.raises(VersionMismatchError):
        pipeline.load()


def test_non_strict_version_mismatch_collects_warning(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    record = _train_one(tmp_path, dataset, split)
    pipeline = ClassificationInferencePipeline(record.artifact_dir, feature_version="999.0.0", strict=False)
    pipeline.load()  # must not raise
    assert pipeline.version_warnings


# --------------------------------------------------------------------------- #
# ClassificationPredictor -- probability sum, explainability, live MarketState
# --------------------------------------------------------------------------- #
def test_predictor_requires_loaded_pipeline() -> None:
    pipeline = ClassificationInferencePipeline.__new__(ClassificationInferencePipeline)
    pipeline.model = None
    pipeline.model_version = None
    with pytest.raises(ValueError):
        ClassificationPredictor(pipeline)


def test_predictor_probabilities_sum_to_one_and_explanation_present(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    record = _train_one(tmp_path, dataset, split)
    pipeline = ClassificationInferencePipeline(record.artifact_dir).load()
    predictor = ClassificationPredictor(pipeline)

    names = dataset.feature_names
    X = dataset.X[split.test_idx][:1]

    class _FakeMarketState:
        def to_vector(self):
            return X.ravel(), names

        def to_dict(self):
            return {f"{n}_valid": 1.0 for n in names[:10]}

    prediction = predictor.predict(_FakeMarketState(), symbol="EUR_USD", timeframe="M5")

    total = prediction.buy_probability + prediction.sell_probability + prediction.no_trade_probability
    assert total == pytest.approx(1.0, abs=1e-6)
    assert_probabilities_sum_to_one(prediction.class_probabilities)  # must not raise

    assert prediction.predicted_class in ("BUY", "SELL", "NO_TRADE")
    assert 0.0 <= prediction.prediction_confidence <= 100.0
    assert 0.0 <= prediction.probability_margin <= 1.0
    assert 0.0 <= prediction.prediction_entropy <= 1.0
    assert prediction.symbol == "EUR_USD" and prediction.timeframe == "M5"
    assert prediction.prediction_horizon == 5

    # Explainability -- every required field, deterministic given the same input.
    assert any(line.startswith("Predicted class:") for line in prediction.explanation)
    assert any("BUY probability" in line for line in prediction.explanation)
    assert any("SELL probability" in line for line in prediction.explanation)
    assert any("NO_TRADE probability" in line for line in prediction.explanation)
    assert any("Prediction confidence" in line for line in prediction.explanation)
    assert any("Probability margin" in line for line in prediction.explanation)
    assert any("Historical accuracy" in line for line in prediction.explanation)
    assert any("Calibration" in line for line in prediction.explanation)
    assert any("Top positive features" in line or "Top negative features" in line for line in prediction.explanation)
    assert any("Most influential market structure signals" in line for line in prediction.explanation)

    prediction2 = predictor.predict(_FakeMarketState(), symbol="EUR_USD", timeframe="M5")
    assert prediction.explanation == prediction2.explanation  # deterministic


def test_predictor_confidence_breakdown_has_all_six_factors(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    record = _train_one(tmp_path, dataset, split, n_bootstrap=5)
    pipeline = ClassificationInferencePipeline(record.artifact_dir).load()
    predictor = ClassificationPredictor(pipeline)
    names = dataset.feature_names
    X = dataset.X[split.test_idx][:1]

    class _FakeMarketState:
        def to_vector(self):
            return X.ravel(), names

        def to_dict(self):
            return {f"{n}_valid": 1.0 for n in names[:10]}

    prediction = predictor.predict(_FakeMarketState())
    breakdown = prediction.confidence_breakdown
    for key in (
        "probability_separation", "historical_accuracy", "distribution_distance",
        "feature_completeness", "prediction_stability", "calibration_quality", "overall",
    ):
        assert key in breakdown
        assert 0.0 <= breakdown[key] <= 100.0


# --------------------------------------------------------------------------- #
# validator.py -- class/horizon/version/schema validation
# --------------------------------------------------------------------------- #
def test_validate_classes_and_horizon_passes_for_matching_request() -> None:
    version = current_classification_version(("SELL", "NO_TRADE", "BUY"), 5)
    validate_classes_and_horizon(version, requested_classes=("SELL", "NO_TRADE", "BUY"), requested_horizon=5)


def test_validate_classes_and_horizon_rejects_class_mismatch() -> None:
    version = current_classification_version(("SELL", "NO_TRADE", "BUY"), 5)
    with pytest.raises(ClassMismatchError):
        validate_classes_and_horizon(version, requested_classes=("DOWN", "UP"))


def test_validate_classes_and_horizon_rejects_horizon_mismatch() -> None:
    version = current_classification_version(("SELL", "NO_TRADE", "BUY"), 5)
    with pytest.raises(ClassMismatchError):
        validate_classes_and_horizon(version, requested_horizon=10)


def test_validate_for_inference_strict_raises_on_schema_mismatch() -> None:
    version_info = current_version_info("1.0.0")
    schema = FeatureSchema.from_feature_names(["a", "b", "c"], version_info)
    with pytest.raises(Exception):
        validate_for_inference(
            model_version_info=version_info, current_version_info=version_info,
            feature_schema=schema, feature_names=["a", "b"], X=np.zeros((1, 2)), strict=True,
        )


def test_validate_for_inference_non_strict_collects_issues() -> None:
    version_info = current_version_info("1.0.0")
    schema = FeatureSchema.from_feature_names(["a", "b", "c"], version_info)
    issues = validate_for_inference(
        model_version_info=version_info, current_version_info=version_info,
        feature_schema=schema, feature_names=["a", "b"], X=np.zeros((1, 2)), strict=False,
    )
    assert issues  # must report the feature-count mismatch, not raise


def test_validate_for_inference_clean_case_returns_no_issues() -> None:
    version_info = current_version_info("1.0.0")
    schema = FeatureSchema.from_feature_names(["a", "b"], version_info)
    issues = validate_for_inference(
        model_version_info=version_info, current_version_info=version_info,
        feature_schema=schema, feature_names=["a", "b"], X=np.zeros((1, 2)), strict=False,
    )
    assert issues == []
