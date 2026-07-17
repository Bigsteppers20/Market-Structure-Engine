"""Tests for model_monitor.retraining_manager -- RETRAINING SAFETY and
MODEL COMPARISON sections: candidate training, comparison, promotion,
archiving, and full run_cycle orchestration across manual/scheduled/adaptive
modes."""
from __future__ import annotations

import numpy as np
import pytest

from model_monitor.calibration_monitor import CalibrationMonitor
from model_monitor.config import MonitorConfig, PromotionPolicy
from model_monitor.drift_detector import DriftDetector
from model_monitor.exceptions import PromotionError, RetrainingError
from model_monitor.health_engine import HealthEngine
from model_monitor.model_registry import ModelLifecycleMetadata, ModelLifecycleRegistry
from model_monitor.prediction_monitor import PredictionLog, PredictionSnapshot, ResolvedPrediction
from model_monitor.retraining_manager import CandidateArtifact, RetrainingManager, compare_models
from model_monitor.retraining_policy import RetrainingRecommendation

NAMES = [f"f{i}" for i in range(5)]
NOW = "2026-07-01T00:00:00+00:00"


def _lifecycle(**overrides) -> ModelLifecycleMetadata:
    base = dict(
        model_name="m", version="1", task_type="regression", status="production",
        training_date="2026-06-25T00:00:00+00:00", training_dataset_size=1000,
        feature_version="1", training_version="1", strategy_version="1", dataset_version="1",
    )
    base.update(overrides)
    return ModelLifecycleMetadata(**base)


def _health_and_drift(config: MonitorConfig):
    rng = np.random.default_rng(0)
    log = PredictionLog()
    for i in range(120):
        pred = rng.normal(0, 0.001)
        actual = pred + rng.normal(0, 0.0002)
        snap = PredictionSnapshot(
            task_type="regression", model_name="m", model_version="1", feature_version="1",
            training_version="1", symbol="EUR_USD", timeframe="M5", prediction_horizon=5,
            timestamp="t", decision_index=i, feature_vector=list(rng.normal(0, 1, 5)), feature_names=NAMES,
            confidence=rng.uniform(40, 90), predicted_value=pred, raw_predictions={},
        )
        log._resolved.append(ResolvedPrediction(snapshot=snap, resolved_at="t", actual_value=actual))
    cal_report = CalibrationMonitor().evaluate(log.resolved)
    detector = DriftDetector(config).fit_baseline(rng.normal(0, 1, (300, 5)), NAMES)
    drift_report = detector.detect(rng.normal(0, 1, (50, 5)), NAMES)
    health_report = HealthEngine(config).evaluate(
        model_name="m", model_version="1", task_type="regression", prediction_log=log,
        drift_report=drift_report, calibration_report=cal_report, lifecycle=_lifecycle(), now_iso=NOW,
    )
    return health_report, drift_report


# --------------------------------------------------------------------------- #
# compare_models -- MODEL COMPARISON
# --------------------------------------------------------------------------- #
def test_compare_models_deploys_clear_improvement() -> None:
    production = CandidateArtifact(version="1", metrics={"rmse": 0.01}, training_date=NOW, training_dataset_size=1000)
    candidate = CandidateArtifact(version="2", metrics={"rmse": 0.005}, training_date=NOW, training_dataset_size=1200)
    result = compare_models(task_type="regression", production=production, candidate=candidate, policy=PromotionPolicy())
    assert result.decision == "deploy"
    assert result.relative_improvement == pytest.approx(0.5)


def test_compare_models_rejects_worse_candidate() -> None:
    production = CandidateArtifact(version="1", metrics={"rmse": 0.01}, training_date=NOW, training_dataset_size=1000)
    candidate = CandidateArtifact(version="2", metrics={"rmse": 0.02}, training_date=NOW, training_dataset_size=1200)
    result = compare_models(task_type="regression", production=production, candidate=candidate, policy=PromotionPolicy())
    assert result.decision == "reject"


def test_compare_models_tie_goes_to_further_evaluation_by_default() -> None:
    production = CandidateArtifact(version="1", metrics={"rmse": 0.01000}, training_date=NOW, training_dataset_size=1000)
    candidate = CandidateArtifact(version="2", metrics={"rmse": 0.010001}, training_date=NOW, training_dataset_size=1200)
    result = compare_models(task_type="regression", production=production, candidate=candidate, policy=PromotionPolicy())
    assert result.decision == "further_evaluation"


def test_compare_models_tie_promotion_allowed_by_policy() -> None:
    production = CandidateArtifact(version="1", metrics={"rmse": 0.01000}, training_date=NOW, training_dataset_size=1000)
    candidate = CandidateArtifact(version="2", metrics={"rmse": 0.010001}, training_date=NOW, training_dataset_size=1200)
    policy = PromotionPolicy(allow_tie_promotion=True)
    result = compare_models(task_type="regression", production=production, candidate=candidate, policy=policy)
    assert result.decision == "deploy"


def test_compare_models_rejects_on_latency_regression_even_if_accuracy_improves() -> None:
    production = CandidateArtifact(
        version="1", metrics={"rmse": 0.01}, training_date=NOW, training_dataset_size=1000, inference_latency_ms=10.0,
    )
    candidate = CandidateArtifact(
        version="2", metrics={"rmse": 0.001}, training_date=NOW, training_dataset_size=1200, inference_latency_ms=100.0,
    )
    result = compare_models(task_type="regression", production=production, candidate=candidate, policy=PromotionPolicy())
    assert result.decision == "reject"
    assert any("latency" in r.lower() for r in result.reasons)


def test_compare_models_classification_uses_balanced_accuracy_by_default() -> None:
    production = CandidateArtifact(version="1", metrics={"balanced_accuracy": 0.5}, training_date=NOW, training_dataset_size=1000)
    candidate = CandidateArtifact(version="2", metrics={"balanced_accuracy": 0.6}, training_date=NOW, training_dataset_size=1200)
    result = compare_models(task_type="classification", production=production, candidate=candidate, policy=PromotionPolicy())
    assert result.primary_metric == "balanced_accuracy"
    assert result.decision == "deploy"


def test_compare_models_no_common_metric_raises() -> None:
    production = CandidateArtifact(version="1", metrics={"custom_metric": 1.0}, training_date=NOW, training_dataset_size=1000)
    candidate = CandidateArtifact(version="2", metrics={"another_metric": 1.0}, training_date=NOW, training_dataset_size=1200)
    with pytest.raises(PromotionError):
        compare_models(task_type="regression", production=production, candidate=candidate, policy=PromotionPolicy())


def test_compare_models_reports_feature_importance_changes() -> None:
    production = CandidateArtifact(
        version="1", metrics={"rmse": 0.01}, training_date=NOW, training_dataset_size=1000,
        feature_importance={"a": 1.0, "b": 2.0},
    )
    candidate = CandidateArtifact(
        version="2", metrics={"rmse": 0.005}, training_date=NOW, training_dataset_size=1200,
        feature_importance={"a": 1.5, "b": 1.0},
    )
    result = compare_models(task_type="regression", production=production, candidate=candidate, policy=PromotionPolicy())
    assert result.feature_importance_changes == {"a": pytest.approx(0.5), "b": pytest.approx(-1.0)}


# --------------------------------------------------------------------------- #
# RetrainingManager.run_cycle
# --------------------------------------------------------------------------- #
def test_manual_mode_notifies_but_never_trains(tmp_path) -> None:
    config = MonitorConfig(retraining_mode="manual", health_threshold=99.9)
    health_report, drift_report = _health_and_drift(config)
    registry = ModelLifecycleRegistry(tmp_path)
    registry.register(_lifecycle())
    manager = RetrainingManager(config, registry)
    recommendation = RetrainingRecommendation(recommended=True, priority="high", reasons=["forced for test"])

    outcome = manager.run_cycle(
        model_name="m", task_type="regression", mode="manual", health_report=health_report,
        drift_report=drift_report, recommendation=recommendation, last_trained_iso="2026-01-01T00:00:00+00:00",
        now_iso=NOW, new_sample_count=1000,
    )
    assert outcome.triggered is False
    assert any(n.type == "retraining_recommended" for n in manager.notifications.history)
    assert registry.get("m").status == "production"  # unchanged


def test_missing_train_fn_raises_when_triggered(tmp_path) -> None:
    config = MonitorConfig(retraining_mode="scheduled")
    health_report, drift_report = _health_and_drift(config)
    registry = ModelLifecycleRegistry(tmp_path)
    registry.register(_lifecycle(training_date="2020-01-01T00:00:00+00:00"))
    manager = RetrainingManager(config, registry)
    recommendation = RetrainingRecommendation(recommended=False, priority="none")

    with pytest.raises(RetrainingError):
        manager.run_cycle(
            model_name="m", task_type="regression", mode="scheduled", health_report=health_report,
            drift_report=drift_report, recommendation=recommendation,
            last_trained_iso="2020-01-01T00:00:00+00:00", now_iso=NOW, new_sample_count=1000,
        )


def test_better_candidate_gets_promoted_and_production_archived(tmp_path) -> None:
    config = MonitorConfig(retraining_mode="scheduled")
    health_report, drift_report = _health_and_drift(config)
    registry = ModelLifecycleRegistry(tmp_path)
    registry.register(_lifecycle(version="1", status="production"))
    manager = RetrainingManager(config, registry)
    recommendation = RetrainingRecommendation(recommended=True, priority="high", reasons=["x"])

    prod_rmse = health_report.performance.historical.metrics["rmse"]

    def train_better() -> CandidateArtifact:
        return CandidateArtifact(version="2", metrics={"rmse": prod_rmse * 0.5}, training_date=NOW, training_dataset_size=1300)

    outcome = manager.run_cycle(
        model_name="m", task_type="regression", mode="scheduled", health_report=health_report,
        drift_report=drift_report, recommendation=recommendation, last_trained_iso="2020-01-01T00:00:00+00:00",
        now_iso=NOW, new_sample_count=1000, train_candidate_fn=train_better,
    )
    assert outcome.triggered is True
    assert outcome.comparison.decision == "deploy"
    assert outcome.promoted_version == "2"
    assert registry.get("m", "1").status == "archived"
    assert registry.get("m", "2").status == "production"
    assert any(n.type == "candidate_promoted" for n in manager.notifications.history)


def test_worse_candidate_gets_rejected_production_unchanged(tmp_path) -> None:
    config = MonitorConfig(retraining_mode="scheduled")
    health_report, drift_report = _health_and_drift(config)
    registry = ModelLifecycleRegistry(tmp_path)
    registry.register(_lifecycle(version="1", status="production"))
    manager = RetrainingManager(config, registry)
    recommendation = RetrainingRecommendation(recommended=True, priority="high", reasons=["x"])

    prod_rmse = health_report.performance.historical.metrics["rmse"]

    def train_worse() -> CandidateArtifact:
        return CandidateArtifact(version="2", metrics={"rmse": prod_rmse * 2.0}, training_date=NOW, training_dataset_size=1300)

    outcome = manager.run_cycle(
        model_name="m", task_type="regression", mode="scheduled", health_report=health_report,
        drift_report=drift_report, recommendation=recommendation, last_trained_iso="2020-01-01T00:00:00+00:00",
        now_iso=NOW, new_sample_count=1000, train_candidate_fn=train_worse,
    )
    assert outcome.triggered is True
    assert outcome.comparison.decision == "reject"
    assert outcome.promoted_version is None
    assert registry.get("m", "1").status == "production"  # never overwritten
    assert registry.get("m", "2").status == "archived"
    assert any(n.type == "candidate_rejected" for n in manager.notifications.history)


def test_first_model_promoted_unconditionally_when_no_production_exists(tmp_path) -> None:
    config = MonitorConfig(retraining_mode="scheduled")
    health_report, drift_report = _health_and_drift(config)
    registry = ModelLifecycleRegistry(tmp_path)
    manager = RetrainingManager(config, registry)
    recommendation = RetrainingRecommendation(recommended=True, priority="high", reasons=["x"])

    def train_first() -> CandidateArtifact:
        return CandidateArtifact(version="1", metrics={"rmse": 0.01}, training_date=NOW, training_dataset_size=500)

    outcome = manager.run_cycle(
        model_name="new_model", task_type="regression", mode="scheduled", health_report=health_report,
        drift_report=drift_report, recommendation=recommendation, last_trained_iso="2020-01-01T00:00:00+00:00",
        now_iso=NOW, new_sample_count=1000, train_candidate_fn=train_first,
    )
    assert outcome.promoted_version == "1"
    assert registry.production_version("new_model").version == "1"


def test_scheduled_mode_does_not_trigger_when_not_due(tmp_path) -> None:
    config = MonitorConfig(retraining_mode="scheduled")
    health_report, drift_report = _health_and_drift(config)
    registry = ModelLifecycleRegistry(tmp_path)
    registry.register(_lifecycle())
    manager = RetrainingManager(config, registry)
    recommendation = RetrainingRecommendation(recommended=False, priority="none")

    outcome = manager.run_cycle(
        model_name="m", task_type="regression", mode="scheduled", health_report=health_report,
        drift_report=drift_report, recommendation=recommendation, last_trained_iso=NOW,
        now_iso=NOW, new_sample_count=0,
    )
    assert outcome.triggered is False
