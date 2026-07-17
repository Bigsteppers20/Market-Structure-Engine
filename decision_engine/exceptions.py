"""Exception hierarchy for the Decision Engine.

Version/schema incompatibilities reuse ``training.versioning``'s exceptions
directly (re-exported here for convenience) rather than duplicating them;
everything below is specific to this engine.
"""
from __future__ import annotations

from training.versioning import SchemaMismatchError, VersionMismatchError

__all__ = [
    "DecisionEngineError",
    "InvalidConfigError",
    "MissingAnalysisError",
    "SchemaMismatchError",
    "VersionMismatchError",
]


class DecisionEngineError(RuntimeError):
    """Base class for every error raised by ``decision_engine``."""


class InvalidConfigError(DecisionEngineError):
    """Raised when ``DecisionEngineConfig`` is misconfigured."""


class MissingAnalysisError(DecisionEngineError):
    """Raised when ``DecisionEngine.decide()`` is called without at least
    one analytical input (a ``StrategyEvaluation`` is always required;
    regression/classification predictions are individually optional but
    at least one of the three must be present)."""
