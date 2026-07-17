"""Logistic Regression Engine: estimates class probabilities for future
trading outcomes from the current ``MarketState``.

Consumes only ``market_structure.MarketState`` -- never raw candles, never
an indicator computed here, never structure/pattern detection duplicated
here, and never Strategy Engine output. Not a trade execution engine and
does not decide BUY/SELL/NO_TRADE; its sole output, ``ClassificationPrediction``,
is meant to be consumed by a future Decision Engine.

Built entirely on the existing platform: ``ml_pipeline`` for dataset
building/label generation/feature preprocessing, ``training`` for the
abstract ``Trainer``/``ModelRegistry``/``InferencePipeline``/versioning/
metrics infrastructure (subclassed and extended, never modified), and
``market_structure`` for the feature vector itself. Deliberately independent
of ``linear_regression`` and ``strategy`` -- three separate analytical
systems whose outputs a future Decision Engine will combine.
"""
from .calibration import (
    CALIBRATION_METHODS,
    brier_score,
    calibrate_estimator,
    compute_calibration_curve,
    expected_calibration_error,
)
from .classification_model import BALANCING_STRATEGIES, ClassificationModel
from .config import DEFAULT_CLASSES, THRESHOLD_STRATEGIES, ClassificationConfig
from .confidence import ConfidenceBreakdown, compute_confidence
from .evaluator import ClassificationEvaluator
from .exceptions import (
    CalibrationError,
    ClassMismatchError,
    InvalidHorizonError,
    InvalidThresholdError,
    LogisticRegressionError,
    ModelNotTrainedError,
    PredictionError,
    SchemaMismatchError,
    UnsupportedBalancingStrategyError,
    UnsupportedClassSetError,
    VersionMismatchError,
)
from .feature_mapper import extract_feature_vector, feature_completeness
from .inference import ClassificationInferencePipeline
from .label_manager import ConfigurableClassificationLabelGenerator
from .live_inference import LiveInferenceResponse, to_live_inference
from .logistic_engine import LogisticRegressionEngine
from .metrics import compute_all_classification_metrics
from .model_health import compute_model_health
from .model_registry import ClassificationModelMetadata, ClassificationModelRegistry
from .predictor import ClassificationPrediction, ClassificationPredictor
from .probability_engine import (
    assert_probabilities_sum_to_one,
    predicted_class,
    prediction_entropy,
    probability_margin,
    to_class_probabilities,
)
from .threshold_manager import ThresholdManager
from .trainer import LogisticRegressionTrainer
from .validator import validate_classes_and_horizon, validate_for_inference
from .version import (
    LOGISTIC_REGRESSION_ENGINE_VERSION,
    ClassificationModelVersion,
    current_classification_version,
)

__version__ = "1.0.0"

__all__ = [
    "LogisticRegressionEngine",
    "ClassificationConfig",
    "DEFAULT_CLASSES",
    "BALANCING_STRATEGIES",
    "THRESHOLD_STRATEGIES",
    "CALIBRATION_METHODS",
    "LogisticRegressionTrainer",
    "ClassificationModel",
    "ConfigurableClassificationLabelGenerator",
    "ClassificationInferencePipeline",
    "ClassificationPredictor",
    "ClassificationPrediction",
    "LiveInferenceResponse",
    "to_live_inference",
    "compute_model_health",
    "ClassificationEvaluator",
    "ClassificationModelRegistry",
    "ClassificationModelMetadata",
    "ThresholdManager",
    "ConfidenceBreakdown",
    "compute_confidence",
    "extract_feature_vector",
    "feature_completeness",
    "compute_all_classification_metrics",
    "calibrate_estimator",
    "compute_calibration_curve",
    "brier_score",
    "expected_calibration_error",
    "to_class_probabilities",
    "predicted_class",
    "probability_margin",
    "prediction_entropy",
    "assert_probabilities_sum_to_one",
    "validate_classes_and_horizon",
    "validate_for_inference",
    "ClassificationModelVersion",
    "LOGISTIC_REGRESSION_ENGINE_VERSION",
    "current_classification_version",
    "LogisticRegressionError",
    "ModelNotTrainedError",
    "UnsupportedClassSetError",
    "UnsupportedBalancingStrategyError",
    "InvalidThresholdError",
    "InvalidHorizonError",
    "PredictionError",
    "CalibrationError",
    "ClassMismatchError",
    "VersionMismatchError",
    "SchemaMismatchError",
]
