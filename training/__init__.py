"""Model-agnostic machine learning training infrastructure.

Sits downstream of ``market_structure`` (feature extraction) and
``ml_pipeline`` (dataset building), and upstream of any concrete model
implementation (linear/logistic regression, random forest, gradient
boosting, XGBoost, LightGBM, deep learning, or anything added later).

This package implements no predictive model itself -- :class:`Trainer` is
abstract. It provides everything a concrete model implementation needs
around it: dataset loading, preprocessing/scaling/selection orchestration,
metrics, evaluation reporting, artifact persistence, experiment tracking,
a model registry, feature/version compatibility enforcement, and an
inference-preparation pipeline that stops just short of calling a model.
"""
from .artifacts import ArtifactManager, ModelPlaceholder
from .config import TrainingConfig
from .evaluator import EvaluationEngine, EvaluationReport
from .experiment import ExperimentManager, ExperimentRecord
from .inference import InferencePipeline
from .metrics import (
    InferenceStatistics,
    Metric,
    TrainingStatistics,
    compute_classification_metrics,
    compute_regression_metrics,
    register_metric,
)
from .registry import ModelMetadata, ModelRegistry
from .trainer import Trainer
from .versioning import (
    FeatureSchema,
    SchemaMismatchError,
    VersionInfo,
    VersionMismatchError,
    assert_schema_compatible,
    current_version_info,
    validate_schema,
    verify_version_compatibility,
)

__version__ = "1.0.0"

__all__ = [
    "Trainer",
    "TrainingConfig",
    "ExperimentManager",
    "ExperimentRecord",
    "ModelRegistry",
    "ModelMetadata",
    "ArtifactManager",
    "ModelPlaceholder",
    "EvaluationEngine",
    "EvaluationReport",
    "InferencePipeline",
    "Metric",
    "TrainingStatistics",
    "InferenceStatistics",
    "compute_regression_metrics",
    "compute_classification_metrics",
    "register_metric",
    "VersionInfo",
    "VersionMismatchError",
    "SchemaMismatchError",
    "FeatureSchema",
    "current_version_info",
    "verify_version_compatibility",
    "validate_schema",
    "assert_schema_compatible",
]
