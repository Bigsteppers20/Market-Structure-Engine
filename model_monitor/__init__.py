"""Model Monitoring and Adaptive Retraining System.

Continuously monitors deployed Linear Regression and Logistic Regression
models (and any future model family that supplies an equivalent adapter):
evaluates health, detects feature/target/residual/regime drift, tracks
calibration, recommends retraining, and -- only when explicitly enabled --
performs adaptive retraining with a train-candidate/compare/promote-or-
archive workflow that never overwrites a production model in place.

Does not execute trades, and does not implement a Decision Engine, Risk
Manager, or the Agentic AI itself -- ``ModelMonitor.to_agentic_report()``
produces the structured dict that a future Agentic AI consumes.

Sits *above* ``linear_regression``/``logistic_regression`` by design (its
entire job is to monitor them), unlike those two engines, which must stay
independent siblings of each other.
"""
from .calibration_monitor import CalibrationMonitor, CalibrationReport, ConfidenceBucket, detect_calibration_drift
from .config import (
    DEFAULT_HEALTH_WEIGHTS,
    RETRAINING_MODES,
    RETRAINING_SCHEDULES,
    MonitorConfig,
    NotificationPolicy,
    PromotionPolicy,
    RetrainingScheduleConfig,
)
from .drift_detector import (
    DistributionShift,
    DriftDetector,
    DriftReport,
    RegimeDriftReport,
    RegimeSnapshot,
    classify_regime,
    detect_distribution_shift,
    detect_regime_drift,
)
from .exceptions import (
    InsufficientDataError,
    InvalidConfigError,
    ModelMonitorError,
    PromotionError,
    RetrainingError,
    SchemaMismatchError,
    UnknownModelError,
    UnresolvedPredictionError,
    VersionMismatchError,
)
from .feature_drift import FeatureDriftDetector, FeatureDriftMetric, FeatureDriftReport
from .health_engine import HealthEngine, ModelHealthReport
from .health_score import HEALTH_SCORE_FACTORS, HealthScoreBreakdown, compute_health_score, severity_to_score
from .model_registry import ModelLifecycleMetadata, ModelLifecycleRegistry
from .monitor import ModelMonitor
from .notification_manager import NOTIFICATION_TYPES, Notification, NotificationManager
from .performance_monitor import PerformanceMonitor, PerformanceReport, RollingHistoricalPerformance
from .prediction_monitor import (
    PredictionLog,
    PredictionSnapshot,
    ResolvedPrediction,
    from_classification_prediction,
    from_regression_prediction,
)
from .retraining_manager import CandidateArtifact, ComparisonResult, RetrainingManager, RetrainingOutcome, compare_models
from .retraining_policy import PRIORITIES, RetrainingRecommendation, evaluate_retraining_policy
from .retraining_scheduler import is_due, should_trigger_retraining
from .validator import (
    assert_valid_prediction_snapshot,
    validate_for_monitoring,
    validate_lifecycle_identity,
    validate_prediction_snapshot,
)
from .version import MODEL_MONITOR_VERSION, MonitoringVersion, current_monitoring_version

__version__ = "1.0.0"

__all__ = [
    "ModelMonitor",
    "MonitorConfig",
    "PromotionPolicy",
    "NotificationPolicy",
    "RetrainingScheduleConfig",
    "RETRAINING_MODES",
    "RETRAINING_SCHEDULES",
    "DEFAULT_HEALTH_WEIGHTS",
    "FeatureDriftDetector",
    "FeatureDriftMetric",
    "FeatureDriftReport",
    "DriftDetector",
    "DriftReport",
    "RegimeSnapshot",
    "RegimeDriftReport",
    "DistributionShift",
    "classify_regime",
    "detect_regime_drift",
    "detect_distribution_shift",
    "PredictionLog",
    "PredictionSnapshot",
    "ResolvedPrediction",
    "from_regression_prediction",
    "from_classification_prediction",
    "CalibrationMonitor",
    "CalibrationReport",
    "ConfidenceBucket",
    "detect_calibration_drift",
    "PerformanceMonitor",
    "PerformanceReport",
    "RollingHistoricalPerformance",
    "HealthScoreBreakdown",
    "HEALTH_SCORE_FACTORS",
    "compute_health_score",
    "severity_to_score",
    "HealthEngine",
    "ModelHealthReport",
    "RetrainingRecommendation",
    "PRIORITIES",
    "evaluate_retraining_policy",
    "is_due",
    "should_trigger_retraining",
    "CandidateArtifact",
    "ComparisonResult",
    "compare_models",
    "RetrainingManager",
    "RetrainingOutcome",
    "NotificationManager",
    "Notification",
    "NOTIFICATION_TYPES",
    "ModelLifecycleMetadata",
    "ModelLifecycleRegistry",
    "validate_prediction_snapshot",
    "assert_valid_prediction_snapshot",
    "validate_for_monitoring",
    "validate_lifecycle_identity",
    "MonitoringVersion",
    "MODEL_MONITOR_VERSION",
    "current_monitoring_version",
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
