"""Forex Dynamics -- single-import convenience bundle over every package this
platform is built from: ``market_structure``, ``ml_pipeline``, ``training``,
``strategy``, ``strategies``, ``linear_regression``, ``logistic_regression``,
``model_monitor``, ``decision_engine``, and ``api``.

This package computes nothing itself. Every name it exposes is the exact
same object as in its home package -- nothing is copied, wrapped, or
reimplemented -- so::

    from forex_dynamics import MarketStructureEngine, DecisionEngine, TrendFollowingStrategy

works instead of importing from several top-level packages individually.

Name collisions are NOT silently resolved
------------------------------------------
Ten names are defined identically in more than one of the underlying
packages but mean different things in each (e.g. every ML/training engine
has its own ``ModelNotTrainedError``, ``VersionMismatchError``,
``SchemaMismatchError``). Blindly chaining ``from x import *`` would let
whichever package is imported last silently shadow the others under the
same bare name -- picking a winner nobody asked for. Instead, those ten
names are deliberately left OUT of this package's flat namespace; import
them from their owning package directly (e.g.
``from linear_regression import ModelNotTrainedError``). See
``_EXCLUDED_COLLISIONS`` below for the full list and every package that
defines each one.
"""
from __future__ import annotations

# Market Structure Engine -- OHLCV -> MarketState feature vector.
from market_structure import *  # noqa: F401,F403
from market_structure import __all__ as _market_structure_all

# ML pipeline -- dataset/splitting utilities shared by both ML engines.
from ml_pipeline import *  # noqa: F401,F403
from ml_pipeline import __all__ as _ml_pipeline_all

# Training infrastructure -- trainer/registry/versioning shared scaffolding.
from training import *  # noqa: F401,F403
from training import __all__ as _training_all

# Strategy Engine -- the rule-based evaluation framework itself.
from strategy import *  # noqa: F401,F403
from strategy import __all__ as _strategy_all

# Concrete Strategy Lab strategies -- strategies/__init__.py defines no
# __all__ of its own (each strategy lives in its own submodule), so import
# every strategy class + its default_config() factory explicitly.
from strategies.ict_strategy import IctStrategy, default_config as ict_default_config
from strategies.london_breakout import LondonBreakoutStrategy, default_config as london_breakout_default_config
from strategies.scalping_strategy import ScalpingStrategy, default_config as scalping_default_config
from strategies.swing_strategy import SwingStrategy, default_config as swing_default_config
from strategies.trend_following import TrendFollowingStrategy, default_config as trend_following_default_config

# Linear Regression Engine.
from linear_regression import *  # noqa: F401,F403
from linear_regression import __all__ as _linear_regression_all

# Logistic Regression Engine.
from logistic_regression import *  # noqa: F401,F403
from logistic_regression import __all__ as _logistic_regression_all

# Model Monitor.
from model_monitor import *  # noqa: F401,F403
from model_monitor import __all__ as _model_monitor_all

# Decision Engine.
from decision_engine import *  # noqa: F401,F403
from decision_engine import __all__ as _decision_engine_all

# api/ -- the FastAPI app instance is the one thing that package produces;
# api/__init__.py itself exposes nothing importable.
from api.main import app as fastapi_app

__version__ = "1.0.0"

#: name -> every owning package, for the 10 names deliberately excluded
#: from __all__ below because they mean different things in each package.
_EXCLUDED_COLLISIONS = {
    "VersionMismatchError": ("training", "linear_regression", "logistic_regression", "model_monitor", "decision_engine"),
    "SchemaMismatchError": ("training", "linear_regression", "logistic_regression", "model_monitor", "decision_engine"),
    "compute_confidence": ("strategy", "linear_regression", "logistic_regression"),
    "ConfidenceBreakdown": ("strategy", "linear_regression", "logistic_regression"),
    "extract_feature_vector": ("linear_regression", "logistic_regression"),
    "feature_completeness": ("linear_regression", "logistic_regression"),
    "ModelNotTrainedError": ("linear_regression", "logistic_regression"),
    "InvalidHorizonError": ("linear_regression", "logistic_regression"),
    "PredictionError": ("linear_regression", "logistic_regression"),
    "InvalidConfigError": ("model_monitor", "decision_engine"),
}

_ALL_SOURCE_NAMES = (
    *_market_structure_all, *_ml_pipeline_all, *_training_all, *_strategy_all,
    *_linear_regression_all, *_logistic_regression_all, *_model_monitor_all, *_decision_engine_all,
)
assert set(_EXCLUDED_COLLISIONS) == {
    n for n in set(_ALL_SOURCE_NAMES) if _ALL_SOURCE_NAMES.count(n) > 1
}, "A source package's __all__ changed -- _EXCLUDED_COLLISIONS above is now out of date."

for _name in _EXCLUDED_COLLISIONS:
    del globals()[_name]

__all__ = sorted(
    {n for n in _ALL_SOURCE_NAMES if n not in _EXCLUDED_COLLISIONS}
    | {
        "IctStrategy", "LondonBreakoutStrategy", "ScalpingStrategy", "SwingStrategy", "TrendFollowingStrategy",
        "ict_default_config", "london_breakout_default_config", "scalping_default_config",
        "swing_default_config", "trend_following_default_config",
        "fastapi_app",
    }
)
