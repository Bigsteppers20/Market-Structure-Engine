"""Exception hierarchy for the Model Monitoring and Adaptive Retraining System.

Version/schema incompatibilities reuse ``training.versioning``'s exceptions
directly (re-exported here for convenience) rather than duplicating them;
everything below is specific to this system.
"""
from __future__ import annotations

from training.versioning import SchemaMismatchError, VersionMismatchError

__all__ = [
    "ModelMonitorError",
    "InvalidConfigError",
    "UnknownModelError",
    "InsufficientDataError",
    "UnresolvedPredictionError",
    "PromotionError",
    "RetrainingError",
    "VersionMismatchError",
    "SchemaMismatchError",
]


class ModelMonitorError(RuntimeError):
    """Base class for every error raised by ``model_monitor``."""


class InvalidConfigError(ModelMonitorError):
    """Raised when ``MonitorConfig`` (or a nested policy) is misconfigured."""


class UnknownModelError(ModelMonitorError):
    """Raised when a monitored model key has no registered lifecycle entry."""


class InsufficientDataError(ModelMonitorError):
    """Raised when an operation needs more resolved predictions / baseline
    samples than are currently available."""


class UnresolvedPredictionError(ModelMonitorError):
    """Raised when an operation requires a resolved outcome but only
    pending (unresolved) predictions are available."""


class PromotionError(ModelMonitorError):
    """Raised when a candidate model cannot be compared/promoted (e.g. task
    type mismatch against the production model)."""


class RetrainingError(ModelMonitorError):
    """Raised when a retraining cycle cannot proceed (e.g. no trainer
    callback supplied for adaptive/scheduled mode)."""
