"""Exception hierarchy for the Logistic Regression Engine.

Version/schema incompatibilities reuse ``training.versioning``'s exceptions
directly (re-exported here for convenience) rather than duplicating them;
everything below is specific to this engine.
"""
from __future__ import annotations

from training.versioning import SchemaMismatchError, VersionMismatchError

__all__ = [
    "LogisticRegressionError",
    "ModelNotTrainedError",
    "UnsupportedClassSetError",
    "InvalidThresholdError",
    "InvalidHorizonError",
    "PredictionError",
    "CalibrationError",
    "ClassMismatchError",
    "UnsupportedBalancingStrategyError",
    "VersionMismatchError",
    "SchemaMismatchError",
]


class LogisticRegressionError(RuntimeError):
    """Base class for every error raised by ``logistic_regression``."""


class ModelNotTrainedError(LogisticRegressionError):
    """Raised when inference is attempted before a model has been trained/loaded."""


class UnsupportedClassSetError(LogisticRegressionError):
    """Raised when ``classes`` is empty, has duplicates, or fewer than 2 entries."""


class InvalidThresholdError(LogisticRegressionError):
    """Raised when a probability threshold or label-generation threshold is out of range."""


class InvalidHorizonError(LogisticRegressionError):
    """Raised when ``prediction_horizon`` is not a positive integer."""


class PredictionError(LogisticRegressionError):
    """Raised when prediction fails for a reason not covered by a version/schema check."""


class CalibrationError(LogisticRegressionError):
    """Raised when an unsupported/misconfigured calibration method is requested."""


class ClassMismatchError(LogisticRegressionError):
    """Raised when a loaded model's class set/horizon doesn't match what was requested."""


class UnsupportedBalancingStrategyError(LogisticRegressionError):
    """Raised when an unknown class-balancing strategy is requested."""
