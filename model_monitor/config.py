"""Configuration for the Model Monitoring and Adaptive Retraining System.

One explicit, serializable dataclass tree, mirroring the same convention
used by ``market_structure.EngineConfig`` / ``training.TrainingConfig`` /
``logistic_regression.ClassificationConfig``: every threshold is explicit
and overridable, nothing is read from global state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

from .exceptions import InvalidConfigError

RETRAINING_SCHEDULES = ("daily", "weekly", "monthly", "custom")
RETRAINING_MODES = ("manual", "scheduled", "adaptive")

#: Default weights for health_score.py's 10 required factors. Every key here
#: must exist in health_score.HEALTH_SCORE_FACTORS -- validated below.
DEFAULT_HEALTH_WEIGHTS: Dict[str, float] = {
    "prediction_accuracy": 0.20,
    "prediction_stability": 0.10,
    "confidence_calibration": 0.10,
    "feature_drift": 0.15,
    "target_drift": 0.08,
    "residual_drift": 0.08,
    "rolling_error": 0.10,
    "market_regime_change": 0.09,
    "training_age": 0.05,
    "prediction_coverage": 0.05,
}


@dataclass(slots=True)
class PromotionPolicy:
    """Candidate-vs-production comparison policy (MODEL COMPARISON section).

    Attributes
    ----------
    min_relative_improvement:
        A candidate must improve the primary metric by at least this
        fraction (e.g. 0.02 = 2%) over production to be promoted outright.
    allow_tie_promotion:
        When True, a candidate within ``tie_tolerance`` of production (no
        clear winner) is still promoted if it also improves calibration or
        stability; when False, ties fall to "further evaluation".
    tie_tolerance:
        Relative difference below which two metric values are considered
        tied.
    max_inference_latency_regression:
        A candidate may be rejected if its inference latency regresses by
        more than this fraction versus production, even if accuracy improves.
    """

    min_relative_improvement: float = 0.02
    allow_tie_promotion: bool = False
    tie_tolerance: float = 0.005
    max_inference_latency_regression: float = 0.5

    def __post_init__(self) -> None:
        if self.min_relative_improvement < 0:
            raise InvalidConfigError("min_relative_improvement must be >= 0.")
        if self.tie_tolerance < 0:
            raise InvalidConfigError("tie_tolerance must be >= 0.")


@dataclass(slots=True)
class NotificationPolicy:
    """Which notifications actually get emitted (NOTIFICATION SYSTEM section).

    Attributes
    ----------
    min_severity:
        One of ``"info"``, ``"warning"``, ``"critical"`` -- notifications
        below this severity are generated (for history) but not dispatched
        to registered handlers.
    """

    min_severity: str = "info"

    def __post_init__(self) -> None:
        if self.min_severity not in ("info", "warning", "critical"):
            raise InvalidConfigError(f"min_severity={self.min_severity!r}, expected info/warning/critical.")


@dataclass(slots=True)
class RetrainingScheduleConfig:
    """SCHEDULED retraining cadence (RETRAINING MODES section).

    ``interval_days`` is derived from ``frequency`` unless ``frequency``
    is ``"custom"``, in which case it must be supplied explicitly.
    """

    frequency: str = "weekly"
    custom_interval_days: Optional[float] = None

    def __post_init__(self) -> None:
        if self.frequency not in RETRAINING_SCHEDULES:
            raise InvalidConfigError(f"frequency={self.frequency!r}, expected one of {RETRAINING_SCHEDULES}.")
        if self.frequency == "custom" and not self.custom_interval_days:
            raise InvalidConfigError("frequency='custom' requires custom_interval_days > 0.")

    @property
    def interval_days(self) -> float:
        return {
            "daily": 1.0, "weekly": 7.0, "monthly": 30.0,
        }.get(self.frequency, float(self.custom_interval_days or 0.0))


@dataclass(slots=True)
class MonitorConfig:
    """Complete configuration for the monitoring/retraining system.

    Attributes
    ----------
    health_threshold:
        Health score (0-100) below which the model is considered degraded.
    feature_drift_threshold:
        Overall feature-drift severity (0-1) above which drift is considered
        significant.
    performance_threshold:
        Per-metric degradation thresholds, e.g.
        ``{"rmse_increase_pct": 0.15, "accuracy_decrease_pct": 0.10}`` --
        a metric moving worse by more than its threshold flags degradation.
    min_new_samples:
        Minimum count of newly *resolved* predictions required before
        retraining is even considered (adaptive/scheduled).
    max_model_age_days:
        Model age above which ``training_age`` health degrades and
        retraining is recommended regardless of other factors.
    auto_retraining_enabled:
        Master switch for adaptive mode -- even if all adaptive conditions
        are met, retraining never auto-triggers when this is False.
    retraining_mode:
        One of ``"manual"``, ``"scheduled"``, ``"adaptive"``.
    retraining_schedule:
        Used when ``retraining_mode == "scheduled"``.
    promotion_policy, notification_policy:
        See their own dataclasses above.
    health_weights:
        Overrides for health_score.py's 10 factor weights (must sum to ~1.0
        and cover exactly the required factor set).
    rolling_window:
        Number of most-recent resolved predictions used for "rolling"
        performance/calibration/drift statistics (vs. "historical" = all-time).
    outlier_z_threshold:
        |z-score| above which a live feature value counts as an outlier
        for feature_drift.py's outlier-frequency diagnostic.
    market_regime_lookback:
        Number of most-recent MarketState snapshots used to characterize
        the *current* market regime for drift_detector.py.
    """

    health_threshold: float = 60.0
    feature_drift_threshold: float = 0.3
    performance_threshold: Dict[str, float] = field(default_factory=lambda: {
        "rmse_increase_pct": 0.15, "mae_increase_pct": 0.15,
        "accuracy_decrease_pct": 0.10, "f1_decrease_pct": 0.10,
    })
    min_new_samples: int = 200
    max_model_age_days: float = 30.0
    auto_retraining_enabled: bool = False
    retraining_mode: str = "manual"
    retraining_schedule: RetrainingScheduleConfig = field(default_factory=RetrainingScheduleConfig)
    promotion_policy: PromotionPolicy = field(default_factory=PromotionPolicy)
    notification_policy: NotificationPolicy = field(default_factory=NotificationPolicy)
    health_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HEALTH_WEIGHTS))
    rolling_window: int = 200
    outlier_z_threshold: float = 3.0
    market_regime_lookback: int = 100

    def __post_init__(self) -> None:
        if not (0.0 <= self.health_threshold <= 100.0):
            raise InvalidConfigError("health_threshold must be in [0, 100].")
        if not (0.0 <= self.feature_drift_threshold <= 1.0):
            raise InvalidConfigError("feature_drift_threshold must be in [0, 1].")
        if self.min_new_samples < 0:
            raise InvalidConfigError("min_new_samples must be >= 0.")
        if self.max_model_age_days <= 0:
            raise InvalidConfigError("max_model_age_days must be > 0.")
        if self.retraining_mode not in RETRAINING_MODES:
            raise InvalidConfigError(f"retraining_mode={self.retraining_mode!r}, expected one of {RETRAINING_MODES}.")
        if self.rolling_window < 1:
            raise InvalidConfigError("rolling_window must be >= 1.")
        if self.outlier_z_threshold <= 0:
            raise InvalidConfigError("outlier_z_threshold must be > 0.")
        expected_factors = set(DEFAULT_HEALTH_WEIGHTS)
        if set(self.health_weights) != expected_factors:
            raise InvalidConfigError(
                f"health_weights must cover exactly {sorted(expected_factors)}, got {sorted(self.health_weights)}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MonitorConfig":
        d = dict(d)
        schedule = d.get("retraining_schedule")
        if isinstance(schedule, dict):
            d["retraining_schedule"] = RetrainingScheduleConfig(**schedule)
        promotion = d.get("promotion_policy")
        if isinstance(promotion, dict):
            d["promotion_policy"] = PromotionPolicy(**promotion)
        notification = d.get("notification_policy")
        if isinstance(notification, dict):
            d["notification_policy"] = NotificationPolicy(**notification)
        return cls(**d)
