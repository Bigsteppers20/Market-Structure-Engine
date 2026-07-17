"""Experiment tracking: one JSON record per training run, capturing every
field the spec requires for full reproducibility/auditability.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import TrainingConfig
from .utils import ensure_dir, get_git_commit, new_id, utc_timestamp


@dataclass(slots=True)
class ExperimentRecord:
    """Everything the spec requires to be recorded for a training run."""

    experiment_id: str
    timestamp: str
    feature_version: str
    dataset_version: str
    strategy_version: str
    scaler_used: str
    feature_selector_used: Optional[str]
    hyperparameters: Dict[str, Any]
    training_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    testing_metrics: Dict[str, Any]
    training_duration_seconds: float
    random_seed: int
    git_commit: Optional[str]
    model_family: str = "unspecified"
    task_type: str = "unspecified"
    experiment_name: str = "unnamed_experiment"
    artifact_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentRecord":
        return cls(**d)


class ExperimentManager:
    """Creates, persists, and retrieves :class:`ExperimentRecord` entries.

    One JSON file per experiment under ``root/experiments/<experiment_id>.json``,
    plus a flat ``root/experiments/index.json`` for fast listing.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = ensure_dir(Path(root) / "experiments")

    def new_record(
        self,
        *,
        config: TrainingConfig,
        model_family: str,
        task_type: str,
        hyperparameters: Dict[str, Any],
        training_metrics: Dict[str, Any],
        validation_metrics: Dict[str, Any],
        testing_metrics: Dict[str, Any],
        training_duration_seconds: float,
        artifact_dir: str | Path,
        repo_dir: Optional[Path] = None,
        experiment_id: Optional[str] = None,
    ) -> ExperimentRecord:
        """Build a fully-populated record for a completed training run.

        Pass ``experiment_id`` explicitly when the caller already generated
        one (e.g. to name an artifact directory before the record exists);
        otherwise a fresh id is minted here.
        """
        return ExperimentRecord(
            experiment_id=experiment_id or new_id("exp"),
            timestamp=utc_timestamp(),
            feature_version=config.feature_version,
            dataset_version=config.dataset_version,
            strategy_version=config.strategy_version,
            scaler_used=config.scaler,
            feature_selector_used=config.feature_selector,
            hyperparameters=dict(hyperparameters),
            training_metrics=dict(training_metrics),
            validation_metrics=dict(validation_metrics),
            testing_metrics=dict(testing_metrics),
            training_duration_seconds=training_duration_seconds,
            random_seed=config.random_seed,
            git_commit=get_git_commit(repo_dir),
            model_family=model_family,
            task_type=task_type,
            experiment_name=config.experiment_name,
            artifact_dir=str(artifact_dir),
        )

    def log(self, record: ExperimentRecord) -> Path:
        """Persist a record and update the index. Returns the record's path."""
        path = self.root / f"{record.experiment_id}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2, default=str), encoding="utf-8")
        self._append_index(record)
        return path

    def load(self, experiment_id: str) -> ExperimentRecord:
        path = self.root / f"{experiment_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No experiment record for {experiment_id!r} at {path}")
        return ExperimentRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_experiments(self) -> List[ExperimentRecord]:
        index_path = self.root / "index.json"
        if not index_path.exists():
            return []
        ids = json.loads(index_path.read_text(encoding="utf-8"))
        return [self.load(exp_id) for exp_id in ids]

    def _append_index(self, record: ExperimentRecord) -> None:
        index_path = self.root / "index.json"
        ids = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
        if record.experiment_id not in ids:
            ids.append(record.experiment_id)
        index_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")
