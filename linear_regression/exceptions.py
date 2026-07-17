"""Exception hierarchy for the Linear Regression Engine.

Version/schema incompatibilities reuse ``training.versioning``'s exceptions
directly (re-exported here for convenience) rather than duplicating them;
everything below is specific to this engine.
"""
from __future__ import annotations

from training.versioning import SchemaMismatchError, VersionMismatchError

__all__ = [
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


class LinearRegressionError(RuntimeError):
    """Base class for every error raised by ``linear_regression``."""


class ModelNotTrainedError(LinearRegressionError):
    """Raised when inference is attempted before a model has been trained/loaded."""


class UnsupportedTargetError(LinearRegressionError):
    """Raised when a configured regression target isn't in the target registry."""


class UnsupportedModelTypeError(LinearRegressionError):
    """Raised when an unknown ``model_type`` is requested for ``RegressionModel``."""


class InvalidHorizonError(LinearRegressionError):
    """Raised when ``prediction_horizon`` is not a positive integer."""


class PredictionError(LinearRegressionError):
    """Raised when prediction fails for a reason not covered by a version/schema check."""


class TargetMismatchError(LinearRegressionError):
    """Raised when a loaded model's regression target/horizon doesn't match what was requested."""
