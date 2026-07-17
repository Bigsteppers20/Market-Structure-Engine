"""Decision Engine: combines the Strategy Engine, Linear Regression Engine,
and Logistic Regression Engine into one unified, explainable
``DecisionResult`` -- the single source of truth for the platform.

Pure orchestration layer: computes no indicator, reads no candle, performs
no market structure analysis, executes no trade, and calculates no risk or
position size (``DecisionResult.position_size`` is a stable placeholder for
the existing Risk Manager to fill in). Every analytical judgment comes from
the three upstream engines' own public outputs.
"""
from .config import DecisionEngineConfig
from .decision_engine import DecisionEngine
from .decision_result import (
    DecisionMetadata,
    DecisionResult,
    Explainability,
    LinearRegressionAnalysis,
    LogisticRegressionAnalysis,
    MarketAnalysis,
    PositionSizePlaceholder,
    StrategyAnalysis,
    StrategyVerdict,
    TradePlan,
)
from .exceptions import (
    DecisionEngineError,
    InvalidConfigError,
    MissingAnalysisError,
    SchemaMismatchError,
    VersionMismatchError,
)
from .validator import (
    assert_valid_decision_result_dict,
    validate_decision_inputs,
    validate_decision_result_dict,
)
from .version import DECISION_ENGINE_VERSION

__version__ = "1.0.0"

__all__ = [
    "DecisionEngine",
    "DecisionEngineConfig",
    "DecisionResult",
    "StrategyAnalysis",
    "LinearRegressionAnalysis",
    "LogisticRegressionAnalysis",
    "TradePlan",
    "PositionSizePlaceholder",
    "MarketAnalysis",
    "StrategyVerdict",
    "Explainability",
    "DecisionMetadata",
    "validate_decision_result_dict",
    "assert_valid_decision_result_dict",
    "validate_decision_inputs",
    "DECISION_ENGINE_VERSION",
    "DecisionEngineError",
    "InvalidConfigError",
    "MissingAnalysisError",
    "VersionMismatchError",
    "SchemaMismatchError",
]
