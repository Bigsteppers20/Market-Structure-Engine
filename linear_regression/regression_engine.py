"""The production entry point: ``MarketState`` in, ``RegressionPrediction`` out.

Live operation, entirely in memory::

    market_state = mse_engine.analyze()              # Market Structure Engine
    prediction = regression_engine.predict(market_state, symbol="EUR_USD", timeframe="M5")

``RegressionEngine`` composes one ``LinearRegressionTrainer`` per configured
target for training (each a full, unmodified reuse of
``training.Trainer.run()``) and one ``RegressionInferencePipeline`` per
target for inference, presenting both as a single facade -- the same role
``strategy.StrategyEngine`` plays for the Strategy Engine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from market_structure import MarketState
from ml_pipeline.dataset_builder import Dataset
from ml_pipeline.splitter import SplitResult
from training.experiment import ExperimentRecord

from .config import RegressionConfig
from .exceptions import ModelNotTrainedError, UnsupportedTargetError
from .inference import RegressionInferencePipeline
from .model_registry import RegressionModelMetadata, RegressionModelRegistry
from .predictor import RegressionPrediction, RegressionPredictor
from .target_generator import REGRESSION_TARGET_REGISTRY, augment_dataset_targets
from .trainer import LinearRegressionTrainer


class RegressionEngine:
    """Trains and/or serves predictions for every target in ``config.targets``."""

    def __init__(self, config: RegressionConfig, repo_dir: Optional[Path] = None) -> None:
        self.config = config
        self.repo_dir = repo_dir
        self.registry = RegressionModelRegistry(config.training_config.output_dir)
        self._trainers: Dict[str, LinearRegressionTrainer] = {}
        self._pipelines: Dict[str, RegressionInferencePipeline] = {}
        self._predictor: Optional[RegressionPredictor] = None

    # ------------------------------------------------------------------ #
    def train(
        self, dataset: Dataset, split: SplitResult, raw_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, ExperimentRecord]:
        """Train one independent linear model per configured target.

        Each target's model is trained via a fresh ``LinearRegressionTrainer``
        (i.e. a full, unmodified pass through ``training.Trainer.run()``),
        registered in :attr:`registry`, and returned as an ``ExperimentRecord``.

        Parameters
        ----------
        raw_df:
            The exact historical DataFrame originally passed to
            ``DatasetBuilder.build()``. Required only if any configured
            target is one of this engine's 5 additions
            (``maximum_favorable_excursion``, ``maximum_adverse_excursion``,
            ``average_future_price``, ``future_range``, ``future_midpoint``)
            that ``ml_pipeline.DatasetBuilder`` cannot compute natively --
            see :func:`target_generator.augment_dataset_targets`.
        """
        dataset = self._ensure_targets_available(dataset, raw_df)
        records: Dict[str, ExperimentRecord] = {}
        for target in self.config.targets:
            trainer = LinearRegressionTrainer(self.config, repo_dir=self.repo_dir)
            record = trainer.run(dataset, split, target_name=target)
            self._trainers[target] = trainer
            records[target] = record

            self.registry.register(RegressionModelMetadata(
                model_name=f"{self.config.training_config.experiment_name}__{target}",
                version=self.config.training_config.strategy_version,
                training_date=record.timestamp,
                training_dataset=self.config.training_config.dataset_version,
                feature_version=self.config.training_config.feature_version,
                regression_target=target,
                prediction_horizon=self.config.prediction_horizon,
                performance_metrics=record.testing_metrics or record.validation_metrics or record.training_metrics,
                supported_symbols=[self.config.symbol],
                supported_timeframes=[self.config.timeframe],
                artifact_dir=record.artifact_dir,
                model_type=self.config.model_type,
                experiment_id=record.experiment_id,
            ))
        return records

    def _ensure_targets_available(self, dataset: Dataset, raw_df: Optional[pd.DataFrame]) -> Dataset:
        """Compute any configured target ``ml_pipeline.DatasetBuilder`` couldn't
        (see :meth:`train`'s ``raw_df`` parameter), extending ``dataset.y_reg``
        in place. A no-op when every configured target is already present."""
        missing = [t for t in self.config.targets if t not in dataset.y_reg]
        if not missing:
            return dataset
        unknown = [t for t in missing if t not in REGRESSION_TARGET_REGISTRY]
        if unknown:
            raise UnsupportedTargetError(f"Unknown regression target(s): {unknown}.")
        if raw_df is None:
            raise UnsupportedTargetError(
                f"Target(s) {missing} are not natively computed by ml_pipeline.DatasetBuilder "
                "(new to this engine) -- pass raw_df=<the DataFrame given to DatasetBuilder.build()> "
                "to RegressionEngine.train() so they can be computed."
            )
        extra = augment_dataset_targets(dataset, raw_df, missing, self.config.prediction_horizon, self.config.pip_size)
        dataset.y_reg.update(extra)
        return dataset

    # ------------------------------------------------------------------ #
    def load(self, artifact_dirs: Dict[str, str], strict: bool = True) -> "RegressionEngine":
        """Load previously-trained artifacts (one directory per target) for inference."""
        pipelines = {
            target: RegressionInferencePipeline(
                path, feature_version=self.config.training_config.feature_version, strict=strict,
            ).load()
            for target, path in artifact_dirs.items()
        }
        self._pipelines = pipelines
        self._predictor = RegressionPredictor(
            pipelines, feature_version=self.config.training_config.feature_version, pip_size=self.config.pip_size,
        )
        return self

    def load_from_records(self, records: Dict[str, ExperimentRecord], strict: bool = True) -> "RegressionEngine":
        """Convenience: load directly from the records :meth:`train` just returned."""
        return self.load({target: record.artifact_dir for target, record in records.items()}, strict=strict)

    # ------------------------------------------------------------------ #
    def predict(self, market_state: MarketState, symbol: str = "UNKNOWN", timeframe: str = "UNKNOWN") -> RegressionPrediction:
        """Estimate future market movement from the current ``MarketState``.

        Never inspects raw candles, computes an indicator, or detects
        structure itself -- every input comes from ``market_state``.
        """
        if self._predictor is None:
            raise ModelNotTrainedError("Call load() or load_from_records() before predict().")
        return self._predictor.predict(market_state, symbol=symbol, timeframe=timeframe)
