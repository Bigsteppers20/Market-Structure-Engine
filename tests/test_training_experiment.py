"""Tests for training.experiment."""
from __future__ import annotations

import pytest

from training.config import TrainingConfig
from training.experiment import ExperimentManager


def test_new_record_populates_every_required_field(tmp_path) -> None:
    cfg = TrainingConfig(
        experiment_name="exp1", feature_version="1.2.0", dataset_version="3.4.0",
        strategy_version="5.6.0", scaler="robust", feature_selector="kbest", random_seed=7,
    )
    mgr = ExperimentManager(tmp_path)
    record = mgr.new_record(
        config=cfg, model_family="stub", task_type="regression",
        hyperparameters={"alpha": 0.1}, training_metrics={"mae": 0.1},
        validation_metrics={"mae": 0.12}, testing_metrics={"mae": 0.13},
        training_duration_seconds=2.5, artifact_dir=tmp_path / "artifacts" / "exp1",
        # tmp_path is deliberately outside any git repo (unlike the cwd this
        # suite runs from, which get_git_commit(None) would otherwise fall
        # back to) -- pass it explicitly so this assertion is deterministic.
        repo_dir=tmp_path,
    )
    assert record.experiment_id.startswith("exp_")
    assert record.feature_version == "1.2.0"
    assert record.dataset_version == "3.4.0"
    assert record.strategy_version == "5.6.0"
    assert record.scaler_used == "robust"
    assert record.feature_selector_used == "kbest"
    assert record.hyperparameters == {"alpha": 0.1}
    assert record.training_metrics == {"mae": 0.1}
    assert record.validation_metrics == {"mae": 0.12}
    assert record.testing_metrics == {"mae": 0.13}
    assert record.training_duration_seconds == 2.5
    assert record.random_seed == 7
    assert record.git_commit is None  # not a git repo
    assert record.timestamp  # non-empty


def test_log_and_load_round_trip(tmp_path) -> None:
    cfg = TrainingConfig(experiment_name="exp2")
    mgr = ExperimentManager(tmp_path)
    record = mgr.new_record(
        config=cfg, model_family="stub", task_type="classification",
        hyperparameters={}, training_metrics={}, validation_metrics={}, testing_metrics={},
        training_duration_seconds=1.0, artifact_dir=tmp_path / "artifacts" / "exp2",
    )
    path = mgr.log(record)
    assert path.exists()

    loaded = mgr.load(record.experiment_id)
    assert loaded == record


def test_load_missing_experiment_raises(tmp_path) -> None:
    mgr = ExperimentManager(tmp_path)
    with pytest.raises(FileNotFoundError):
        mgr.load("exp_does_not_exist")


def test_list_experiments_returns_all_logged(tmp_path) -> None:
    cfg = TrainingConfig()
    mgr = ExperimentManager(tmp_path)
    ids = []
    for i in range(3):
        record = mgr.new_record(
            config=cfg, model_family="stub", task_type="regression",
            hyperparameters={}, training_metrics={}, validation_metrics={}, testing_metrics={},
            training_duration_seconds=0.1, artifact_dir=tmp_path / f"artifacts/exp{i}",
        )
        mgr.log(record)
        ids.append(record.experiment_id)

    all_records = mgr.list_experiments()
    assert {r.experiment_id for r in all_records} == set(ids)


def test_explicit_experiment_id_is_respected(tmp_path) -> None:
    cfg = TrainingConfig()
    mgr = ExperimentManager(tmp_path)
    record = mgr.new_record(
        config=cfg, model_family="stub", task_type="regression",
        hyperparameters={}, training_metrics={}, validation_metrics={}, testing_metrics={},
        training_duration_seconds=0.1, artifact_dir=tmp_path / "artifacts/expX",
        experiment_id="exp_explicit_id",
    )
    assert record.experiment_id == "exp_explicit_id"
