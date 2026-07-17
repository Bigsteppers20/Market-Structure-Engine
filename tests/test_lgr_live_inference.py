"""Tests for logistic_regression.live_inference -- the minimal production
LIVE INFERENCE contract.

Covers: exact 8-key contract shape (nothing more), predicted value always
in {BUY, SELL, WAIT}, the NO_TRADE -> WAIT mapping (and BUY/SELL
passthrough), model_health populated by the trainer, and two regression
guards confirming evaluation output and the full ClassificationPrediction
object are both untouched by this change.
"""
from __future__ import annotations

import pytest

from logistic_regression.config import ClassificationConfig
from logistic_regression.inference import ClassificationInferencePipeline
from logistic_regression.live_inference import LiveInferenceResponse, to_live_inference
from logistic_regression.predictor import ClassificationPrediction, ClassificationPredictor
from logistic_regression.trainer import LogisticRegressionTrainer
from training.config import TrainingConfig

_LIVE_CONTRACT_KEYS = {
    "prediction", "prediction_confidence", "prediction_horizon",
    "model_version", "feature_version", "training_version", "model_health", "timestamp",
}


def _train_one(tmp_path, dataset, split, **overrides):
    tc = TrainingConfig(experiment_name="lgr_live_inference_test", output_root=str(tmp_path), random_seed=3)
    base = dict(prediction_horizon=5, n_bootstrap=6, training_config=tc)
    base.update(overrides)
    cfg = ClassificationConfig(**base)
    trainer = LogisticRegressionTrainer(cfg)
    record = trainer.run(dataset, split)
    return record


class _FakeMarketState:
    def __init__(self, X_row, names):
        self._X_row = X_row
        self._names = names

    def to_vector(self):
        return self._X_row.ravel(), self._names

    def to_dict(self):
        return {f"{n}_valid": 1.0 for n in self._names[:10]}


def _base_prediction(**overrides) -> ClassificationPrediction:
    base = dict(
        buy_probability=0.1, sell_probability=0.1, no_trade_probability=0.8,
        predicted_class="NO_TRADE", prediction_confidence=83.0, probability_margin=0.6,
        prediction_entropy=0.3, model_version="1.0.0", feature_version="1.0.0",
        training_version="1.0.0", timestamp="2026-07-15T00:00:00+00:00",
        symbol="EUR_USD", timeframe="M5", prediction_horizon=5, model_health=94.0,
    )
    base.update(overrides)
    return ClassificationPrediction(**base)


# --------------------------------------------------------------------------- #
# Pure derivation -- hand-built ClassificationPrediction, no training needed
# --------------------------------------------------------------------------- #
def test_no_trade_maps_to_wait() -> None:
    response = to_live_inference(_base_prediction(predicted_class="NO_TRADE"))
    assert response.prediction == "WAIT"


def test_buy_passes_through() -> None:
    response = to_live_inference(_base_prediction(predicted_class="BUY"))
    assert response.prediction == "BUY"


def test_sell_passes_through() -> None:
    response = to_live_inference(_base_prediction(predicted_class="SELL"))
    assert response.prediction == "SELL"


def test_to_dict_has_exactly_the_contract_keys_and_nothing_else() -> None:
    response = to_live_inference(_base_prediction())
    d = response.to_dict()
    assert set(d) == _LIVE_CONTRACT_KEYS
    # Explicitly confirm no evaluation-oriented keys leak through.
    for forbidden in (
        "classes", "buy_probability", "sell_probability", "no_trade_probability",
        "class_probabilities", "confusion_matrix", "accuracy", "precision", "recall",
        "f1", "roc_auc", "pr_auc", "log_loss", "brier_score", "explanation", "warnings",
        "confidence_breakdown", "probability_margin", "prediction_entropy",
    ):
        assert forbidden not in d, f"{forbidden!r} must not appear in the live inference contract"


def test_to_dict_rounds_confidence_and_health_to_int() -> None:
    response = to_live_inference(_base_prediction(prediction_confidence=90.6, model_health=93.5))
    d = response.to_dict()
    assert d["prediction_confidence"] == 91
    assert d["model_health"] == 94
    assert isinstance(d["model_health"], int)


def test_model_health_none_is_preserved_as_none() -> None:
    response = to_live_inference(_base_prediction(model_health=None))
    assert response.to_dict()["model_health"] is None


# --------------------------------------------------------------------------- #
# End-to-end against a real trained model
# --------------------------------------------------------------------------- #
def test_live_inference_from_real_trained_model(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    record = _train_one(tmp_path, dataset, split)
    pipeline = ClassificationInferencePipeline(record.artifact_dir).load()
    predictor = ClassificationPredictor(pipeline)
    names = dataset.feature_names

    seen_predictions = set()
    for i in split.test_idx[:15]:
        X = dataset.X[i:i + 1]
        prediction = predictor.predict(_FakeMarketState(X, names), symbol="EUR_USD", timeframe="M5")
        response = to_live_inference(prediction)
        seen_predictions.add(response.prediction)
        d = response.to_dict()
        assert set(d) == _LIVE_CONTRACT_KEYS
        assert d["prediction"] in ("BUY", "SELL", "WAIT")
        assert isinstance(d["prediction_confidence"], int)
        assert 0 <= d["prediction_confidence"] <= 100
        assert d["prediction_horizon"] == 5
        assert d["model_version"] and d["feature_version"] and d["training_version"]
        assert d["model_health"] is None or 0 <= d["model_health"] <= 100

    assert seen_predictions <= {"BUY", "SELL", "WAIT"}


def test_model_health_populated_by_trainer(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    record = _train_one(tmp_path, dataset, split)
    pipeline = ClassificationInferencePipeline(record.artifact_dir).load()
    assert pipeline.model is not None
    assert pipeline.model.model_health_ is not None
    assert 0.0 <= pipeline.model.model_health_ <= 100.0


# --------------------------------------------------------------------------- #
# Regression guards -- evaluation output and the full prediction object
# --------------------------------------------------------------------------- #
def test_full_classification_prediction_still_has_every_original_field_plus_model_health(
    tmp_path, lgr_dataset_and_split,
) -> None:
    dataset, split, _ = lgr_dataset_and_split
    record = _train_one(tmp_path, dataset, split)
    pipeline = ClassificationInferencePipeline(record.artifact_dir).load()
    predictor = ClassificationPredictor(pipeline)
    names = dataset.feature_names
    X = dataset.X[split.test_idx][:1]

    prediction = predictor.predict(_FakeMarketState(X, names), symbol="EUR_USD", timeframe="M5")
    d = prediction.to_dict()
    for key in (
        "buy_probability", "sell_probability", "no_trade_probability", "predicted_class",
        "prediction_confidence", "probability_margin", "prediction_entropy", "model_version",
        "feature_version", "training_version", "timestamp", "symbol", "timeframe",
        "prediction_horizon", "class_probabilities", "confidence_breakdown", "explanation",
        "warnings", "model_health",
    ):
        assert key in d, f"full ClassificationPrediction.to_dict() lost the {key!r} key"
    assert d["model_health"] is None or 0.0 <= d["model_health"] <= 100.0


def test_training_evaluation_metrics_untouched(tmp_path, lgr_dataset_and_split) -> None:
    """Regression guard for this deliverable: training/evaluation output
    must keep exposing every evaluation metric, even though live inference
    no longer does."""
    dataset, split, _ = lgr_dataset_and_split
    record = _train_one(tmp_path, dataset, split)
    for metrics in (record.testing_metrics, record.validation_metrics):
        for key in (
            "accuracy", "precision", "recall", "f1", "confusion_matrix",
            "roc_auc", "pr_auc", "log_loss", "brier_score",
        ):
            assert key in metrics, f"{key!r} missing from evaluation metrics -- evaluation output regressed"
