"""Configuration for the training infrastructure.

Same convention as ``market_structure.EngineConfig`` and
``ml_pipeline.DatasetConfig``: one explicit, serializable dataclass, nothing
read from global/environment state except where the caller opts in.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from market_structure import EngineConfig
from ml_pipeline import DatasetConfig


@dataclass(slots=True)
class TrainingConfig:
    """Tunable parameters for a training run.

    Attributes
    ----------
    experiment_name:
        Human-readable label for the experiment (stored, not used for logic).
    feature_version, dataset_version, strategy_version:
        Free-form version strings the caller controls, stamped into every
        experiment record and model registry entry. These are distinct from
        the auto-derived engine/builder/pipeline versions in
        :mod:`training.versioning` -- they track *your* feature-set/dataset
        snapshot/trading-strategy iteration, not code versions.
    scaler:
        One of ``"standard"``, ``"minmax"``, ``"robust"``, ``"none"`` --
        forwarded verbatim to ``ml_pipeline.FeatureScaler``.
    feature_selector:
        Optional name forwarded to ``ml_pipeline.FeatureSelector``, or
        ``None`` to keep every feature.
    feature_selector_kwargs:
        Keyword arguments forwarded to ``ml_pipeline.FeatureSelector``.
    output_root:
        Directory under which experiments/artifacts/registry are written.
    random_seed:
        Seed applied via ``training.utils.set_random_seed`` before every run.
    strict_version_check:
        When True (default), any version/schema mismatch during inference
        raises; when False, mismatches are collected as warnings instead
        (useful for controlled migration testing only).
    supported_timeframes, supported_symbols:
        Declared applicability of the resulting model, stored in the
        registry -- purely descriptive, not enforced during training.
    training_strategy:
        Free-form label (e.g. ``"trend_following_v1"``) stored in the
        registry alongside the model.
    """

    experiment_name: str = "unnamed_experiment"
    feature_version: str = "1.0.0"
    dataset_version: str = "1.0.0"
    strategy_version: str = "1.0.0"
    scaler: str = "standard"
    feature_selector: Optional[str] = None
    feature_selector_kwargs: Dict[str, Any] = field(default_factory=dict)
    feature_pipeline_config: DatasetConfig = field(default_factory=DatasetConfig)
    """Passed straight to ``ml_pipeline.FeaturePipeline`` for missing-value
    imputation / encoding -- reuses the dataset builder's own config object
    rather than duplicating its fields here."""
    output_root: str = "training_output"
    random_seed: int = 42
    strict_version_check: bool = True
    supported_timeframes: List[str] = field(default_factory=list)
    supported_symbols: List[str] = field(default_factory=list)
    training_strategy: str = "unspecified"

    def __post_init__(self) -> None:
        if self.scaler not in ("standard", "minmax", "robust", "none"):
            raise ValueError(f"Invalid scaler {self.scaler!r}.")
        if self.random_seed < 0:
            raise ValueError("random_seed must be >= 0.")

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown TrainingConfig field(s): {sorted(unknown)}")
        d = dict(d)
        fp = d.get("feature_pipeline_config")
        if isinstance(fp, dict):
            fp = dict(fp)
            engine = fp.get("engine_config")
            if isinstance(engine, dict):
                # Note: EngineConfig's tuple-typed fields (ema_periods, session_*)
                # round-trip through JSON as lists, not tuples -- functionally
                # equivalent (dataclasses don't enforce field types at runtime)
                # but not identical via `==`. Not corrected here to keep this
                # round-trip simple; construct EngineConfig directly if exact
                # tuple fidelity matters for your use case.
                fp["engine_config"] = EngineConfig(**engine)
            d["feature_pipeline_config"] = DatasetConfig(**fp)
        return cls(**d)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root)
