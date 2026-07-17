"""Strategy Engine: evaluates live ``MarketState`` against configurable
trading strategies.

Consumes only ``market_structure.MarketState`` -- never raw candles, never
the broker, never a hand-computed indicator. Does not execute trades and
does not implement any machine learning; its sole output,
``StrategyEvaluation``, is meant to be consumed by a future Linear/Logistic
Regression model, Decision Engine, Risk Manager, or the existing Agentic AI.
"""
from .compliance import compute_compliance
from .confidence import ConfidenceBreakdown, compute_confidence
from .config import StrategyConfig
from .explanation import generate_explanations
from .rule_base import BUILTIN_RULES, Rule, RuleResult, RuleStatus
from .rule_engine import RuleEngine, RuleSpec, count_by_status
from .scoring import ScoreBreakdown, compute_scores
from .strategy_base import (
    MSE_COMPLIANCE_THRESHOLD,
    MarketBias,
    StrategyEvaluation,
    TradeRecommendation,
    TradingStrategy,
    compute_market_bias,
    compute_mse_compliance,
    compute_recommendation,
)
from .strategy_engine import StrategyEngine
from .strategy_loader import StrategyLoader
from .strategy_registry import StrategyRegistry, default_registry
from .strategy_version import RULE_LIBRARY_VERSION, StrategyVersion, bump_version
from .validator import (
    DuplicateRuleError,
    MarketStateValidationError,
    StrategyValidationError,
    ThresholdValidationError,
    WeightValidationError,
    validate_market_state,
    validate_strategy_config,
)

__version__ = "1.0.0"

__all__ = [
    "StrategyEngine",
    "TradingStrategy",
    "StrategyEvaluation",
    "MarketBias",
    "TradeRecommendation",
    "compute_market_bias",
    "compute_mse_compliance",
    "compute_recommendation",
    "MSE_COMPLIANCE_THRESHOLD",
    "StrategyConfig",
    "StrategyRegistry",
    "default_registry",
    "StrategyLoader",
    "Rule",
    "RuleResult",
    "RuleStatus",
    "BUILTIN_RULES",
    "RuleEngine",
    "RuleSpec",
    "count_by_status",
    "compute_compliance",
    "compute_confidence",
    "ConfidenceBreakdown",
    "compute_scores",
    "ScoreBreakdown",
    "generate_explanations",
    "StrategyVersion",
    "RULE_LIBRARY_VERSION",
    "bump_version",
    "StrategyValidationError",
    "WeightValidationError",
    "DuplicateRuleError",
    "ThresholdValidationError",
    "MarketStateValidationError",
    "validate_strategy_config",
    "validate_market_state",
]
