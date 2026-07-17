"""Linear Regression Engine: estimates future market movement from the
current ``MarketState``.

Consumes only ``market_structure.MarketState`` -- never raw candles, never
an indicator computed here, never structure/pattern detection duplicated
here. Not a trade execution engine and does not decide BUY/SELL; its sole
output, ``RegressionPrediction``, is meant to be consumed by a future
Decision Engine.

Built entirely on the existing platform: ``ml_pipeline`` for dataset
building/feature preprocessing, ``training`` for the abstract ``Trainer``/
``ModelRegistry``/``InferencePipeline``/versioning/metrics infrastructure
(subclassed and extended, never modified), and ``market_structure`` for the
feature vector itself.
"""
from .config import RegressionConfig
from .confidence import ConfidenceBreakdown, compute_confidence
from .evaluator import RegressionEvaluator
from .exceptions import (
    InvalidHorizonError,
    LinearRegressionError,
    ModelNotTrainedError,
    PredictionError,
    SchemaMismatchError,
    TargetMismatchError,
    UnsupportedModelTypeError,
    UnsupportedTargetError,
    VersionMismatchError,
)
from .feature_mapper import extract_feature_vector, feature_completeness
from .inference import RegressionInferencePipeline
from .metrics import compute_all_regression_metrics, prediction_error_distribution, residual_statistics
from .model_registry import RegressionModelMetadata, RegressionModelRegistry
from .predictor import RegressionPrediction, RegressionPredictor
from .regression_engine import RegressionEngine
from .regression_model import RegressionModel
from .target_generator import REGRESSION_TARGET_REGISTRY, TARGET_TO_PREDICTION_FIELD, compute_target, compute_targets
from .trainer import LinearRegressionTrainer
from .version import LINEAR_REGRESSION_ENGINE_VERSION, RegressionModelVersion, current_regression_version

__version__ = "1.0.0"

__all__ = [
    "RegressionEngine",
    "RegressionConfig",
    "LinearRegressionTrainer",
    "RegressionModel",
    "RegressionInferencePipeline",
    "RegressionPredictor",
    "RegressionPrediction",
    "RegressionEvaluator",
    "RegressionModelRegistry",
    "RegressionModelMetadata",
    "ConfidenceBreakdown",
    "compute_confidence",
    "extract_feature_vector",
    "feature_completeness",
    "compute_all_regression_metrics",
    "residual_statistics",
    "prediction_error_distribution",
    "REGRESSION_TARGET_REGISTRY",
    "TARGET_TO_PREDICTION_FIELD",
    "compute_target",
    "compute_targets",
    "RegressionModelVersion",
    "LINEAR_REGRESSION_ENGINE_VERSION",
    "current_regression_version",
    "LinearRegressionError",
    "ModelNotTrainedError",
    "UnsupportedTargetError",
    "UnsupportedModelTypeError",
    "InvalidHorizonError",
    "PredictionError",
    "TargetMismatchError",
    "VersionMismatchError",
    "SchemaMismatchError",
]
