"""The production entry point: ``MarketState`` in, ``ClassificationPrediction`` out.

Live operation, entirely in memory::

    market_state = mse_engine.analyze()              # Market Structure Engine
    prediction = logistic_engine.predict(market_state, symbol="EUR_USD", timeframe="M5")

``LogisticRegressionEngine`` trains one multi-class model (via
``LogisticRegressionTrainer``, a full, unmodified reuse of
``training.Trainer.run()``) and serves it via one
``ClassificationInferencePipeline`` -- the same role
``linear_regression.RegressionEngine`` and ``strategy.StrategyEngine`` play
for their respective engines, kept fully independent of both.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from market_structure import MarketState
from ml_pipeline.dataset_builder import Dataset
from ml_pipeline.splitter import SplitResult
from training.experiment import ExperimentRecord

from .config import ClassificationConfig
from .exceptions import ModelNotTrainedError
from .inference import ClassificationInferencePipeline
from .model_registry import ClassificationModelMetadata, ClassificationModelRegistry
from .predictor import ClassificationPrediction, ClassificationPredictor
from .trainer import LogisticRegressionTrainer


class LogisticRegressionEngine:
    """Trains and/or serves probability predictions for one classifier."""

    def __init__(self, config: ClassificationConfig, repo_dir: Optional[Path] = None) -> None:
        self.config = config
        self.repo_dir = repo_dir
        self.registry = ClassificationModelRegistry(config.training_config.output_dir)
        self._trainer: Optional[LogisticRegressionTrainer] = None
        self._pipeline: Optional[ClassificationInferencePipeline] = None
        self._predictor: Optional[ClassificationPredictor] = None

    # ------------------------------------------------------------------ #
    def train(self, dataset: Dataset, split: SplitResult) -> ExperimentRecord:
        """Train the classifier via a full, unmodified pass through
        ``training.Trainer.run()`` (through ``LogisticRegressionTrainer``),
        register it, and return the ``ExperimentRecord``."""
        trainer = LogisticRegressionTrainer(self.config, repo_dir=self.repo_dir)
        record = trainer.run(dataset, split)
        self._trainer = trainer

        self.registry.register(ClassificationModelMetadata(
            model_name=self.config.training_config.experiment_name,
            version=self.config.training_config.strategy_version,
            training_date=record.timestamp,
            feature_version=self.config.training_config.feature_version,
            training_dataset=self.config.training_config.dataset_version,
            classification_labels=self.config.classes,
            prediction_horizon=self.config.prediction_horizon,
            calibration_method=self.config.calibration_method,
            performance_metrics=record.testing_metrics or record.validation_metrics or record.training_metrics,
            supported_symbols=[self.config.symbol],
            supported_timeframes=[self.config.timeframe],
            artifact_dir=record.artifact_dir,
            experiment_id=record.experiment_id,
        ))
        return record

    # ------------------------------------------------------------------ #
    def load(self, artifact_dir: str | Path, strict: bool = True) -> "LogisticRegressionEngine":
        """Load a previously-trained model for inference."""
        pipeline = ClassificationInferencePipeline(
            artifact_dir, feature_version=self.config.training_config.feature_version, strict=strict,
        ).load()
        self._pipeline = pipeline
        self._predictor = ClassificationPredictor(pipeline)
        return self

    def load_from_record(self, record: ExperimentRecord, strict: bool = True) -> "LogisticRegressionEngine":
        return self.load(record.artifact_dir, strict=strict)

    # ------------------------------------------------------------------ #
    def predict(self, market_state: MarketState, symbol: str = "UNKNOWN", timeframe: str = "UNKNOWN") -> ClassificationPrediction:
        """Estimate class probabilities from the current ``MarketState``.

        Never inspects raw candles, computes an indicator, or detects
        structure itself -- every input comes from ``market_state``. Never
        consumes Strategy Engine output.
        """
        if self._predictor is None:
            raise ModelNotTrainedError("Call load() or load_from_record() before predict().")
        return self._predictor.predict(market_state, symbol=symbol, timeframe=timeframe)
