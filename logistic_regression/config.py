"""Configuration for the Logistic Regression Engine.

Embeds ``training.TrainingConfig`` rather than duplicating any of its
fields -- the same nesting discipline used by every other engine on this
platform. Deliberately imports nothing from ``strategy``/``strategies`` or
``linear_regression``: this engine reuses only the shared platform layer
(``market_structure``, ``ml_pipeline``, ``training``), never a peer
analytical engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

from training.config import TrainingConfig

from .exceptions import (
    InvalidHorizonError,
    InvalidThresholdError,
    UnsupportedBalancingStrategyError,
    UnsupportedClassSetError,
)
from .calibration import CALIBRATION_METHODS

BALANCING_STRATEGIES = ("none", "class_weight", "oversample", "undersample", "balanced_sampling")
THRESHOLD_STRATEGIES = ("argmax", "optimized", "custom")

#: The spec's default 3-class set. The architecture supports any ordered
#: tuple of >= 2 class names (e.g. adding STRONG_BUY/WEAK_BUY/EXIT_LONG/...)
#: without any change to this dataclass or the public API -- see
#: label_manager.py and predictor.py's class_probabilities field.
DEFAULT_CLASSES: Tuple[str, ...] = ("SELL", "NO_TRADE", "BUY")


@dataclass(slots=True)
class ClassificationConfig:
    """Complete configuration for training/running the classifier.

    Attributes
    ----------
    classes:
        Ordered class names. Must have >= 2 entries, no duplicates. Default
        is the spec's ``(SELL, NO_TRADE, BUY)``; extend freely (see module
        docstring).
    prediction_horizon:
        Bars ahead the label looks -- fully configurable, never hardcoded.
    min_pip_movement, min_expected_return, risk_reward_threshold,
    max_adverse_excursion_pips:
        Forwarded to :class:`label_manager.ConfigurableClassificationLabelGenerator`.
    pip_size:
        Price units per pip (0.0001 for most FX pairs, 0.01 for JPY pairs).
    class_balancing:
        One of ``"none"``, ``"class_weight"``, ``"oversample"``,
        ``"undersample"``, ``"balanced_sampling"``.
    calibration_method:
        One of ``"none"``, ``"platt"``, ``"isotonic"``.
    threshold_strategy:
        One of ``"argmax"`` (default), ``"optimized"`` (fit per-class
        thresholds maximizing F1 on an internal holdout), ``"custom"`` (use
        ``custom_thresholds`` verbatim).
    custom_thresholds:
        ``class_name -> probability``, used when ``threshold_strategy == "custom"``.
    n_bootstrap:
        Number of bootstrap-resampled classifiers fit alongside the primary
        one, used for prediction-stability confidence and calibration
        diagnostics (0 disables).
    """

    classes: Tuple[str, ...] = DEFAULT_CLASSES
    prediction_horizon: int = 5
    min_pip_movement: float = 5.0
    min_expected_return: float = 0.0
    risk_reward_threshold: float = 1.5
    max_adverse_excursion_pips: Optional[float] = None
    pip_size: float = 0.0001
    class_balancing: str = "none"
    calibration_method: str = "none"
    threshold_strategy: str = "argmax"
    custom_thresholds: Dict[str, float] = field(default_factory=dict)
    n_bootstrap: int = 10
    model_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    symbol: str = "*"
    timeframe: str = "*"
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        self.classes = tuple(self.classes)
        if len(self.classes) < 2:
            raise UnsupportedClassSetError(f"classes must have >= 2 entries, got {self.classes!r}.")
        if len(set(self.classes)) != len(self.classes):
            raise UnsupportedClassSetError(f"classes contains duplicate(s): {self.classes!r}.")
        if self.prediction_horizon < 1:
            raise InvalidHorizonError(f"prediction_horizon must be >= 1, got {self.prediction_horizon!r}.")
        if self.class_balancing not in BALANCING_STRATEGIES:
            raise UnsupportedBalancingStrategyError(
                f"class_balancing={self.class_balancing!r}, expected one of {BALANCING_STRATEGIES}."
            )
        if self.calibration_method not in CALIBRATION_METHODS:
            raise InvalidThresholdError(f"calibration_method={self.calibration_method!r}, expected one of {CALIBRATION_METHODS}.")
        if self.threshold_strategy not in THRESHOLD_STRATEGIES:
            raise InvalidThresholdError(f"threshold_strategy={self.threshold_strategy!r}, expected one of {THRESHOLD_STRATEGIES}.")
        if self.n_bootstrap < 0:
            raise ValueError("n_bootstrap must be >= 0.")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["classes"] = list(self.classes)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassificationConfig":
        d = dict(d)
        tc = d.get("training_config")
        if isinstance(tc, dict):
            d["training_config"] = TrainingConfig.from_dict(tc)
        if "classes" in d:
            d["classes"] = tuple(d["classes"])
        return cls(**d)
