"""Configuration for the Linear Regression Engine.

Embeds ``training.TrainingConfig`` (which itself embeds
``ml_pipeline.DatasetConfig``, which embeds
``market_structure.EngineConfig``) rather than duplicating any of its
fields -- the same nesting discipline already used throughout this
platform.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from training.config import TrainingConfig

from .exceptions import InvalidHorizonError, UnsupportedModelTypeError, UnsupportedTargetError

MODEL_TYPES = ("linear", "ridge", "lasso", "elasticnet")

#: Every horizon named in the spec's example; any positive int is accepted,
#: this is just the documented default set for UI/validation convenience.
COMMON_HORIZONS = (1, 3, 5, 10, 20, 50)


@dataclass(slots=True)
class RegressionConfig:
    """Complete configuration for training/running one (or more) regression targets.

    Attributes
    ----------
    targets:
        One or more names from
        :data:`linear_regression.target_generator.REGRESSION_TARGET_REGISTRY`.
        A single entry => single-output regression; more than one => the
        engine trains one independent model per target and composes their
        predictions into one :class:`~linear_regression.predictor.RegressionPrediction`
        (see ``regression_engine.py`` module docstring for why -- shared
        preprocessing, independent per-target models, zero duplicated
        ``training.Trainer`` orchestration).
    prediction_horizon:
        Bars ahead each target is computed for -- fully configurable, never
        hardcoded (1/3/5/10/20/50 are the documented common choices).
    model_type:
        One of ``"linear"``, ``"ridge"``, ``"lasso"``, ``"elasticnet"``.
    model_hyperparameters:
        Forwarded to the underlying scikit-learn estimator
        (e.g. ``{"alpha": 1.0}`` for ridge/lasso/elasticnet).
    n_bootstrap:
        Number of bootstrap-resampled estimators fit alongside the primary
        one, used to estimate per-prediction variance/interval (0 disables).
    training_config:
        The full, reused ``training.TrainingConfig`` (scaler, feature
        selector, output paths, random seed, ...).
    """

    targets: List[str] = field(default_factory=lambda: ["next_close"])
    prediction_horizon: int = 5
    model_type: str = "linear"
    model_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    n_bootstrap: int = 10
    pip_size: float = 0.0001
    symbol: str = "*"
    timeframe: str = "*"
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    enable_cross_validation: bool = False
    """When True, ``LinearRegressionTrainer`` additionally runs a walk-forward
    cross-validation pass (see ``cross_validation.py``) on the training
    split and folds the result into the model's ``cross_validation_score``/
    ``target_reliability`` metadata. Defaults to False so existing training
    runs keep their exact current cost/behavior -- this is strictly opt-in."""
    cv_n_folds: int = 5
    """Walk-forward folds used when ``enable_cross_validation`` is True."""

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise UnsupportedModelTypeError(f"model_type={self.model_type!r}, expected one of {MODEL_TYPES}.")
        if self.prediction_horizon < 1:
            raise InvalidHorizonError(f"prediction_horizon must be >= 1, got {self.prediction_horizon!r}.")
        if not self.targets:
            raise UnsupportedTargetError("targets is empty -- configure at least one regression target.")
        if self.n_bootstrap < 0:
            raise ValueError("n_bootstrap must be >= 0.")
        if self.cv_n_folds < 2:
            raise ValueError("cv_n_folds must be >= 2.")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegressionConfig":
        d = dict(d)
        tc = d.get("training_config")
        if isinstance(tc, dict):
            d["training_config"] = TrainingConfig.from_dict(tc)
        return cls(**d)
