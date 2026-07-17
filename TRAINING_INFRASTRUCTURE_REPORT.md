# Training Infrastructure Report

Model-agnostic machine learning training infrastructure (`training/`) built on top
of the existing, unmodified **Market Structure Engine** (`market_structure/`) and
**Dataset Builder / Feature Pipeline** (`ml_pipeline/`). This package implements
**no machine learning algorithm** -- `Trainer` is abstract. Everything below was
verified end-to-end with a pair of trivial test-only stub trainers (a mean-baseline
regressor and a majority-class classifier -- see `tests/_training_stub.py`), which
exist solely to exercise this orchestration logic and are not shipped as part of the
public API.

**Test suite: 206 tests passing across all three packages** (83 of them new, for
`training/` specifically), 1 skipped, 0 failing.

## Architecture Diagram

```mermaid
flowchart LR
    subgraph existing["Existing (unmodified)"]
        MSE["market_structure\nMarketStructureEngine\n185-dim feature_vector()"]
        MLP["ml_pipeline\nDatasetBuilder / FeaturePipeline\nFeatureScaler / FeatureSelector\nTimeSeriesSplitter"]
        MSE --> MLP
    end

    subgraph training["training/ (this deliverable)"]
        CFG["config.py\nTrainingConfig"]
        VER["versioning.py\nVersionInfo / FeatureSchema\nVersionMismatchError"]
        TRN["trainer.py\nTrainer (abstract)"]
        MET["metrics.py\nMetric registry"]
        EVA["evaluator.py\nEvaluationEngine"]
        ART["artifacts.py\nArtifactManager"]
        EXP["experiment.py\nExperimentManager"]
        REG["registry.py\nModelRegistry"]
        INF["inference.py\nInferencePipeline"]
        UTL["utils.py"]

        CFG --> TRN
        VER --> TRN
        UTL --> TRN
        TRN --> MET --> EVA
        TRN --> ART
        TRN --> EXP
        TRN --> REG
        VER --> INF
        ART --> INF
    end

    subgraph future["Future (not built here)"]
        M1["LinearRegressionTrainer"]
        M2["LogisticRegressionTrainer"]
        M3["RandomForestTrainer"]
        M4["XGBoostTrainer"]
        M5["LightGBMTrainer"]
        M6["DeepLearningTrainer"]
        M7["... anything else"]
    end

    MLP -- "Dataset(X, y_reg, y_cls, metadata)" --> TRN
    M1 & M2 & M3 & M4 & M5 & M6 & M7 -. "subclass, implement\n4 abstract hooks" .-> TRN
    INF -- "processed feature vector\n(no predict() call)" --> M1 & M2 & M3 & M4 & M5 & M6 & M7
```

## Class Diagram

```mermaid
classDiagram
    class Trainer {
        <<abstract>>
        +TrainingConfig config
        +model_family: str  *abstract property*
        +task_type: str  *abstract property*
        +build_model(hyperparameters) Any  *abstract*
        +fit_model(model, X_train, y_train) Any  *abstract*
        +predict(model, X) ndarray  *abstract*
        +predict_proba(model, X) ndarray?  optional hook
        +feature_importance(model, names) dict?  optional hook
        +load_dataset(path) Dataset
        +run(dataset, split, target_name) ExperimentRecord
    }
    class TrainingConfig {
        +experiment_name: str
        +feature_version, dataset_version, strategy_version: str
        +scaler: str
        +feature_selector: str?
        +feature_pipeline_config: DatasetConfig
        +random_seed: int
        +to_json() / from_json()
    }
    class VersionInfo {
        +feature_version: str
        +schema_version: str
        +market_structure_engine_version: str
        +dataset_builder_version: str
        +training_pipeline_version: str
    }
    class FeatureSchema {
        +feature_names: List~str~
        +feature_count: int
        +version_info: VersionInfo
        +fingerprint: str
    }
    class ArtifactManager {
        +save_bundle(...) Dict~str,Path~
        +load_bundle() Dict~str,Any~
        +save_joblib/load_joblib()
        +save_pickle/load_pickle()
        +save_json/load_json()
    }
    class ExperimentManager {
        +new_record(...) ExperimentRecord
        +log(record) Path
        +load(id) ExperimentRecord
        +list_experiments() List
    }
    class ExperimentRecord {
        +experiment_id, timestamp
        +feature/dataset/strategy_version
        +scaler_used, feature_selector_used
        +hyperparameters
        +training/validation/testing_metrics
        +training_duration_seconds
        +random_seed, git_commit
    }
    class ModelRegistry {
        +register(metadata) Path
        +get(name, version?) ModelMetadata
        +list_models() / list_versions()
    }
    class ModelMetadata {
        +model_name, version, training_date
        +feature_count, feature_schema_version
        +training_dataset_version
        +performance_metrics
        +supported_timeframes/symbols
        +training_strategy
    }
    class EvaluationEngine {
        +evaluate(...) EvaluationReport
    }
    class EvaluationReport {
        +dataset_summary, feature_summary
        +training_statistics
        +validation/testing_metrics
        +confusion_matrix  real-or-placeholder
        +residual_analysis  real-or-placeholder
        +calibration  real-or-placeholder
        +feature_importance  real-or-placeholder
    }
    class InferencePipeline {
        +load() InferencePipeline
        +prepare(X, names) (ndarray, List~str~)
        +version_warnings, schema_warnings
    }
    class Metric {
        <<abstract>>
        +name: str
        +task_type: str
        +compute(y_true, y_pred) float
    }

    Trainer --> TrainingConfig
    Trainer --> ExperimentManager
    Trainer --> ModelRegistry
    Trainer --> ArtifactManager
    Trainer --> EvaluationEngine
    Trainer ..> FeatureSchema : creates at train time
    ExperimentManager --> ExperimentRecord
    ModelRegistry --> ModelMetadata
    EvaluationEngine --> EvaluationReport
    EvaluationEngine --> Metric
    InferencePipeline --> ArtifactManager
    InferencePipeline --> FeatureSchema : validates against
    FeatureSchema --> VersionInfo
```

## Artifact Flow

Every `Trainer.run()` call writes one directory:
`<output_root>/artifacts/<experiment_id>/`, containing:

| File | Format | Written from |
|---|---|---|
| `scaler.joblib` | joblib | The `ml_pipeline.FeatureScaler` fit on the train split |
| `feature_selector.joblib` | joblib | The `ml_pipeline.FeatureSelector` fit on the train split (only if configured) |
| `feature_pipeline.joblib` | joblib | The `ml_pipeline.FeaturePipeline` fit on the train split (imputation medians, etc.) |
| `config.json` | JSON | `TrainingConfig.to_dict()` |
| `metadata.json` | JSON | experiment id, model family, task type, target name, processed feature names |
| `training_report.json` | JSON | `TrainingStatistics` (duration, sample counts, seed, timestamps) |
| `evaluation_report.json` | JSON | Full `EvaluationReport` (see Evaluation Engine below) |
| `feature_schema.json` | JSON | `FeatureSchema` (raw engine feature names/order/count + `VersionInfo`) -- **this is what inference checks against** |
| `model_placeholder.joblib` | joblib | `ModelPlaceholder(model_family, task_type)` -- stands in for a real fitted model, which this package never produces |

Alongside artifacts, two more directories accumulate across every run under
the same `output_root`:

- `experiments/<experiment_id>.json` + `experiments/index.json` -- the full
  `ExperimentRecord` log (`ExperimentManager`).
- `models/<name>__<version>.json` + `models/index.json` -- the model registry
  (`ModelRegistry`), one entry per `model_name@version`.

`ArtifactManager` also exposes raw `save_pickle`/`load_pickle` for any
artifact type, satisfying the spec's explicit "Joblib and Pickle" requirement
independently of the joblib-by-default bundle path.

## Versioning Strategy

Every artifact bundle carries a `VersionInfo` with **five independently
tracked version numbers**:

1. `feature_version` -- caller-controlled, tracks *your* feature-configuration
   iteration (set via `TrainingConfig.feature_version`).
2. `schema_version` -- `training.versioning.FEATURE_SCHEMA_VERSION`, bumped
   when the training infrastructure's expectations about feature schema shape
   change.
3. `market_structure_engine_version` -- `market_structure.__version__`,
   captured automatically.
4. `dataset_builder_version` -- `ml_pipeline.__version__`, captured
   automatically.
5. `training_pipeline_version` -- `training.versioning.TRAINING_PIPELINE_VERSION`.

`current_version_info()` builds this from whatever is installed *right now*.
`verify_version_compatibility(expected, actual)` compares two `VersionInfo`
instances field-by-field and raises `VersionMismatchError` naming every
field that differs -- not just the first one, so a caller sees the full
picture in one error.

Feature **schema** (names, order, count, dtype) is a separate but related
concern, checked by `validate_schema()`/`assert_schema_compatible()`, which
raises `SchemaMismatchError` -- **a subclass of `VersionMismatchError`**, so
callers can catch either broadly (`except VersionMismatchError`) or
specifically (`except SchemaMismatchError`). A fast fingerprint
(`FeatureSchema.fingerprint`, a SHA-256 hash of the ordered feature names)
allows a cheap equality pre-check before a full diff.

**Enforcement point**: `InferencePipeline.load()` calls
`verify_version_compatibility()` immediately after reading
`feature_schema.json`; `InferencePipeline.prepare()` calls
`assert_schema_compatible()` against the actual incoming feature names before
any transform runs. `Trainer.run()` also accepts an optional
`expected_schema` to block *training itself* from proceeding against a
dataset built with an incompatible feature set.

## Experiment Lifecycle

```mermaid
sequenceDiagram
    participant Caller
    participant Trainer
    participant FP as FeaturePipeline/Scaler/Selector
    participant Eval as EvaluationEngine
    participant Art as ArtifactManager
    participant Exp as ExperimentManager
    participant Reg as ModelRegistry

    Caller->>Trainer: run(dataset, split, target_name, hyperparameters)
    Trainer->>Trainer: set_random_seed(config.random_seed)
    Trainer->>Trainer: build FeatureSchema from dataset.feature_names
    Trainer->>Trainer: select y (y_reg[target_name] or y_cls)
    Trainer->>FP: fit_transform(train) / transform(val, test)
    Trainer->>Trainer: build_model() -> fit_model() [abstract hooks]
    Trainer->>Trainer: predict() on train/val/test [abstract hook]
    Trainer->>Trainer: compute_regression_/classification_metrics()
    Trainer->>Eval: evaluate(task_type, predictions, stats, ...)
    Eval-->>Trainer: EvaluationReport
    Trainer->>Art: save_bundle(scaler, selector, pipeline, config,\nmetadata, reports, schema, model_placeholder)
    Trainer->>Exp: new_record(...) + log(record)
    Trainer->>Reg: register(ModelMetadata)
    Trainer-->>Caller: ExperimentRecord
```

## Inference Lifecycle

```mermaid
sequenceDiagram
    participant Caller
    participant Inf as InferencePipeline
    participant Art as ArtifactManager
    participant Ver as versioning.py
    participant Model as "future model"

    Caller->>Inf: InferencePipeline(artifact_dir, feature_version).load()
    Inf->>Art: load_json("feature_schema")
    Art-->>Inf: FeatureSchema
    Inf->>Ver: verify_version_compatibility(schema.version_info, current)
    alt versions mismatch (strict=True)
        Ver-->>Caller: raise VersionMismatchError
    end
    Inf->>Art: load_joblib(scaler / feature_selector / feature_pipeline)
    Caller->>Inf: prepare(X_raw, feature_names)
    Inf->>Ver: assert_schema_compatible(schema, feature_names, X_raw)
    alt schema mismatch
        Ver-->>Caller: raise SchemaMismatchError
    end
    Inf->>Inf: feature_pipeline.transform() -> scaler.transform() -> selector.transform()
    Inf-->>Caller: (X_processed, feature_names_processed)
    Note over Caller,Model: Caller passes X_processed to a real\nmodel.predict() -- InferencePipeline never does.
```

## Future Extension Points

Every one of these is additive -- no file in `training/` needs to change:

| To add... | Do this |
|---|---|
| A new model family (e.g. XGBoost) | Subclass `Trainer`, implement `model_family`, `task_type`, `build_model`, `fit_model`, `predict` (and optionally `predict_proba`/`feature_importance`). Nothing else changes. |
| A new metric | Subclass `Metric`, call `register_metric(MyMetric())` -- it's picked up automatically by `compute_regression_metrics`/`compute_classification_metrics` (see `tests/test_training_metrics.py::test_register_metric_extends_without_modifying_existing_code`, which does exactly this). |
| A new feature selector | Add a case to `ml_pipeline.FeatureSelector` (already supports `variance`/`correlation`/`mutual_info`/`rfe`/`kbest`) -- `Trainer` already forwards `config.feature_selector`/`feature_selector_kwargs` verbatim. |
| A new export format | Add a method to `ml_pipeline.DatasetExporter`, or a new `save_x`/`load_x` pair on `ArtifactManager` following the existing joblib/pickle/json pattern. |
| A new classification labeling rule | Subclass `ml_pipeline.LabelGenerator` (upstream, in the Dataset Builder) -- `training/` is agnostic to how `y_cls` was produced. |
| Real confusion matrix / residuals / calibration / feature importance | Already wired: `EvaluationEngine` computes these for real the moment predictions/probabilities/importances are supplied. A concrete `Trainer` subclass gets them "for free" by implementing `predict`/`predict_proba`/`feature_importance` -- `evaluator.py` itself never needs to change. |
| A hosted/remote model registry | `ModelRegistry` is file-based by design (local, dependency-free); swap it for a subclass or a different implementation behind the same `register`/`get`/`list_models` interface without touching `Trainer`. |

## What Was Deliberately Not Built

Per the task's explicit scope: no `LinearRegressionTrainer`, no
`LogisticRegressionTrainer`, no Random Forest / XGBoost / LightGBM / deep
learning integration, and no `predict()` call anywhere in `InferencePipeline`.
The only place a "model" object appears in test code is the pair of trivial
stubs in `tests/_training_stub.py` (mean-baseline / majority-class), used
purely to prove the orchestration logic works -- neither is a named ML
algorithm, registered in `training/__init__.py`, or reachable outside the
test suite.
