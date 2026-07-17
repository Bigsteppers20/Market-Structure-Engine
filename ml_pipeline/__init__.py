"""ML Dataset Builder for the Market Structure Engine.

Converts historical OHLCV market data into supervised-learning-ready
datasets (feature matrix X, regression targets y_reg, classification
targets y_cls, metadata) by driving :class:`market_structure.MarketStructureEngine`
over a leakage-safe rolling window, one window per historical candle.

This package never trains, fits, or evaluates a predictive model -- its
responsibility ends when a dataset is ready for a separate training step.
"""
from .config import DatasetConfig
from .dataset_builder import Dataset, DatasetBuilder
from .exporter import DatasetExporter
from .feature_pipeline import FeaturePipeline
from .feature_selector import FeatureSelector
from .label_generator import LabelGenerator, ThresholdLabelGenerator
from .scaler import FeatureScaler
from .splitter import SplitResult, TimeSeriesSplitter
from .validator import ValidationReport, validate_dataset, validate_input_data

__version__ = "1.0.0"

__all__ = [
    "DatasetConfig",
    "Dataset",
    "DatasetBuilder",
    "DatasetExporter",
    "FeaturePipeline",
    "FeatureSelector",
    "LabelGenerator",
    "ThresholdLabelGenerator",
    "FeatureScaler",
    "SplitResult",
    "TimeSeriesSplitter",
    "ValidationReport",
    "validate_dataset",
    "validate_input_data",
]
