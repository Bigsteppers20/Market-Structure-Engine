"""Runs Phases 4-6 and 8 of the Linear Regression Engine improvement spec
against real OANDA historical data and writes
``LINEAR_REGRESSION_IMPROVEMENT_REPORT.md`` -- genuine before/after
generalization evidence, not a maximized single-split R^2.

    Before  -- the engine's true out-of-box default: one linear model per
               target, one 70/15/15 split (``RegressionConfig`` defaults).
    Phase 5 -- ``cross_validation.compare_model_types`` picks the best of
               linear/ridge/lasso/elasticnet per target on that same split.
    Phase 4 -- ``cross_validation.cross_validate`` (walk-forward expanding,
               5 folds, never shuffled) using the Phase-5 winner -- the
               "after" number, and a deliberately *different, harder*
               evidence source than the single-split "before" number (see
               the Methodology section this script writes).
    (final)  -- the Phase-5 winner is retrained through the full production
               pipeline (``RegressionEngine.train()``, CV enabled) -- this
               is the actual "improved" deployable model, and the one whose
               Phase 1/2/3 diagnostics and Phase 8 metadata get reported.
    Phase 6 -- ``target_suitability`` aggregates all of the above into a
               per-target suitability verdict + explanation.
    Phase 8 -- the improved models' ``health_scores_`` plus one real live
               prediction's Decision Engine metadata.

Reuses ``generate_regression_diagnostics_report``'s render functions for
the Phase 1/2/3 reports (regenerated here against the *improved* models,
superseding the earlier ridge/single-split run) rather than duplicating
~150 lines of markdown rendering.

Usage (from the project root, with a configured .env)::

    .venv\\Scripts\\python.exe scripts\\generate_regression_improvement_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "scripts"))

from market_structure import EngineConfig, MarketStructureEngine  # noqa: E402
from ml_pipeline import DatasetBuilder, TimeSeriesSplitter  # noqa: E402
from ml_pipeline.config import DatasetConfig as MLDatasetConfig  # noqa: E402
from ml_pipeline.dataset_builder import Dataset  # noqa: E402
from ml_pipeline.label_generator import REGRESSION_REGISTRY as NATIVE_REGISTRY  # noqa: E402
from ml_pipeline.splitter import SplitResult  # noqa: E402
from oanda_client import BASE_URL, PRACTICE, fetch_candles  # noqa: E402
from training.config import TrainingConfig  # noqa: E402
from training.experiment import ExperimentRecord  # noqa: E402

from linear_regression import RegressionConfig, RegressionEngine  # noqa: E402
from linear_regression.cross_validation import (  # noqa: E402
    CrossValidationResult,
    ModelComparisonResult,
    compare_model_types,
    cross_validate,
    recommend_best_model,
)
from linear_regression.feature_diagnostics import FeatureAuditReport, audit_features_for_target  # noqa: E402
from linear_regression.inference import RegressionInferencePipeline  # noqa: E402
from linear_regression.residual_diagnostics import ResidualDiagnostics, analyze_residuals  # noqa: E402
from linear_regression.target_diagnostics import analyze_all_targets  # noqa: E402
from linear_regression.target_suitability import TargetSuitability, assess_target_suitability

import generate_regression_diagnostics_report as diag  # noqa: E402

OUTPUT_ROOT = ROOT / "lr_output"
NAMED_TARGETS = diag.NAMED_TARGETS
MODEL_TYPES = ("linear", "ridge", "lasso", "elasticnet")
CV_N_FOLDS = 5
_fmt = diag._fmt


# --------------------------------------------------------------------------- #
# Data + baseline ("before")
# --------------------------------------------------------------------------- #
def build_dataset(symbol: str, timeframe: str, count: int, window_size: int, horizon: int, stride: int):
    candles = fetch_candles(symbol, timeframe, count=count)
    print(f"Fetched {len(candles)} live candles ({candles['timestamp'].iloc[0]} -> {candles['timestamp'].iloc[-1]})")
    native_targets = [t for t in NAMED_TARGETS if t in NATIVE_REGISTRY]
    ml_cfg = MLDatasetConfig(
        window_size=window_size, horizon=horizon, stride=stride, symbol=symbol, timeframe=timeframe,
        regression_targets=native_targets,
    )
    dataset: Dataset = DatasetBuilder(ml_cfg).build(candles)
    print(f"Built {len(dataset)} samples x {dataset.X.shape[1]} features")
    split: SplitResult = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(dataset))[0]
    print(f"Split -> train={len(split.train_idx)} val={len(split.val_idx)} test={len(split.test_idx)}")
    return candles, dataset, split


def train_baseline(
    dataset: Dataset, split: SplitResult, candles, symbol: str, timeframe: str, horizon: int,
) -> Dict[str, ExperimentRecord]:
    """The engine's true out-of-box default: ``model_type="linear"``, one
    simple split, no cross-validation. Also performs the MFE/MAE
    augmentation on ``dataset.y_reg`` in place (via ``raw_df=candles``), so
    every later step that reads ``dataset.y_reg[target]`` directly sees all
    8 named targets."""
    training_config = TrainingConfig(
        experiment_name="lr_improvement_before", feature_version="1.0.0", dataset_version="1.0.0",
        strategy_version="1.0.0", output_root=str(OUTPUT_ROOT), scaler="standard", random_seed=42,
    )
    regression_config = RegressionConfig(
        targets=NAMED_TARGETS, prediction_horizon=horizon, model_type="linear",
        model_hyperparameters={}, n_bootstrap=0, pip_size=0.0001,
        symbol=symbol, timeframe=timeframe, training_config=training_config,
    )
    engine = RegressionEngine(regression_config)
    print(f"\n[Before] Training {len(NAMED_TARGETS)} default linear models (single split, no CV)...")
    records = engine.train(dataset, split, raw_df=candles)
    for target, record in records.items():
        m = record.testing_metrics
        print(f"  {target:<28} test R2={m['r2']:+.4f} test RMSE={m['rmse']:.6f}")
    return records


# --------------------------------------------------------------------------- #
# Phase 5 (model selection) + Phase 4 (walk-forward CV)
# --------------------------------------------------------------------------- #
def select_and_validate(
    dataset: Dataset, split: SplitResult, target: str,
) -> tuple[List[ModelComparisonResult], str, Dict[str, float], Optional[CrossValidationResult]]:
    comparison = compare_model_types(
        dataset.X, dataset.y_reg[target], split, target_name=target, model_types=MODEL_TYPES,
    )
    best_type = recommend_best_model(comparison)
    best_result = next(r for r in comparison if r.model_type == best_type)
    cv_result: Optional[CrossValidationResult] = None
    try:
        cv_result = cross_validate(
            dataset.X, dataset.y_reg[target], target_name=target, model_type=best_type,
            method="walk_forward_expanding", n_folds=CV_N_FOLDS,
        )
    except ValueError as exc:
        print(f"  [warn] walk-forward CV skipped for {target!r}: {exc}")
    return comparison, best_type, dict(best_result.hyperparameters), cv_result


# --------------------------------------------------------------------------- #
# Final improved model per target
# --------------------------------------------------------------------------- #
def train_improved(
    dataset: Dataset, split: SplitResult, candles, target: str, best_type: str,
    hyperparameters: Dict[str, float], symbol: str, timeframe: str, horizon: int,
) -> ExperimentRecord:
    training_config = TrainingConfig(
        experiment_name=f"lr_improvement_after_{target}", feature_version="1.0.0", dataset_version="1.0.0",
        strategy_version="1.0.0", output_root=str(OUTPUT_ROOT), scaler="standard", random_seed=42,
    )
    regression_config = RegressionConfig(
        targets=[target], prediction_horizon=horizon, model_type=best_type,
        model_hyperparameters=hyperparameters, n_bootstrap=15, pip_size=0.0001,
        symbol=symbol, timeframe=timeframe, training_config=training_config,
        enable_cross_validation=True, cv_n_folds=CV_N_FOLDS,
    )
    engine = RegressionEngine(regression_config)
    records = engine.train(dataset, split, raw_df=candles)
    return records[target]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_report(
    *, before: Dict[str, ExperimentRecord], comparisons: Dict[str, List[ModelComparisonResult]],
    best_types: Dict[str, str], cv_results: Dict[str, Optional[CrossValidationResult]],
    improved: Dict[str, ExperimentRecord], improved_pipelines: Dict[str, RegressionInferencePipeline],
    suitability: Dict[str, TargetSuitability], live_prediction: Optional[Dict[str, Any]],
    n_samples: int, n_features: int,
) -> str:
    lines: List[str] = [
        "# Linear Regression Improvement Report",
        "",
        "Phases 4-6 and 8 of the Linear Regression Engine improvement spec, run "
        "against real OANDA EUR/USD M5 data by "
        "`scripts/generate_regression_improvement_report.py`. Phases 1-3 and 7 "
        "were already complete (see `TARGET_ANALYSIS_REPORT.md`, "
        "`FEATURE_DIAGNOSTICS_REPORT.md`, `RESIDUAL_ANALYSIS_REPORT.md` -- "
        "regenerated by this run against the *improved* models -- and "
        "`linear_regression/confidence.py`'s 7-factor confidence engine).",
        "",
        "## Executive Summary",
        "",
        f"- Dataset: {n_samples} samples x {n_features} features.",
        f"- {sum(1 for t in suitability.values() if t.verdict == 'suitable')}/{len(suitability)} targets "
        "are suitable for a linear model at their walk-forward cross-validated R^2; "
        f"{sum(1 for t in suitability.values() if t.verdict == 'poor')} are flagged poor -- see Target Suitability below.",
        "- The goal was genuine generalization, not a maximized R^2: every 'after' number below is a "
        "walk-forward cross-validated mean (5 folds, never shuffled), a strictly harder and more honest "
        "estimate than the single-split 'before' number it is compared against.",
        "",
        "## Methodology",
        "",
        "- **Before**: `RegressionConfig` defaults -- `model_type=\"linear\"`, one 70/15/15 time-ordered "
        "split, no cross-validation. This is exactly what a user gets from this engine out of the box.",
        "- **Phase 5 (model selection)**: `cross_validation.compare_model_types` trains linear/ridge/lasso/"
        "elasticnet on the *same* before-split and picks the best by test R^2 (ties by lower RMSE).",
        "- **Phase 4 (cross-validation)**: `cross_validation.cross_validate` runs walk-forward **expanding** "
        f"folds ({CV_N_FOLDS} folds) with the Phase-5 winning model type -- hyperparameters are re-tuned "
        "per fold on an internal, time-ordered holdout carved from that fold's own training data (never "
        "the fold's test data). This is a genuinely different evidence source than the before number: "
        "repeated out-of-sample estimates across time, not one lucky/unlucky split.",
        "- **Improved model**: the Phase-5 winner (type + tuned hyperparameters) is retrained through the "
        "full production pipeline (`RegressionEngine.train()`, cross-validation enabled) -- this is the "
        "actual deployable model behind every Phase 1/2/3/8 result below.",
        "- Never shuffles data at any step (`ml_pipeline.TimeSeriesSplitter`, reused unmodified).",
        "",
        "## Performance Comparison (Before vs. After)",
        "",
        "| Target | Before R2 | Before RMSE | Phase 5 winner | After (CV mean R2 +/- std) | After CV RMSE |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for target in NAMED_TARGETS:
        b = before[target].testing_metrics
        cv = cv_results.get(target)
        cv_r2 = f"{cv.mean_r2:+.4f} +/- {cv.std_r2:.4f}" if cv else "n/a"
        cv_rmse = _fmt(cv.mean_rmse) if cv else "n/a"
        lines.append(
            f"| {target} | {b['r2']:+.4f} | {_fmt(b['rmse'])} | {best_types[target]} | {cv_r2} | {cv_rmse} |"
        )

    lines += ["", "## Model Selection Detail (Phase 5)", ""]
    for target in NAMED_TARGETS:
        lines += [
            f"### `{target}` (winner: **{best_types[target]}**)",
            "",
            "| Model | R2 | MAE | RMSE | Explained Var | Train (ms) | Infer (ms) | Nonzero coef | Hyperparameters |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in comparisons[target]:
            lines.append(
                f"| {r.model_type} | {r.r2:+.4f} | {_fmt(r.mae)} | {_fmt(r.rmse)} | {_fmt(r.explained_variance)} | "
                f"{r.training_time_ms:.2f} | {r.inference_time_ms:.3f} | {r.n_nonzero_coefficients} | {r.hyperparameters} |"
            )
        lines.append("")

    lines += ["## Cross-Validation Detail (Phase 4)", ""]
    for target in NAMED_TARGETS:
        cv = cv_results.get(target)
        if cv is None:
            lines += [f"### `{target}`", "", "Skipped -- insufficient data for the requested fold count.", ""]
            continue
        lines += [
            f"### `{target}` -- {cv.method}, {len(cv.folds)} folds",
            "",
            f"Mean R2={cv.mean_r2:+.4f} (std={cv.std_r2:.4f}), Mean MAE={_fmt(cv.mean_mae)} (std={_fmt(cv.std_mae)}), "
            f"Mean RMSE={_fmt(cv.mean_rmse)} (std={_fmt(cv.std_rmse)})",
            "",
            "| Fold | n_train | n_test | R2 | MAE | RMSE | Explained Var |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for f in cv.folds:
            lines.append(
                f"| {f.fold_index} | {f.n_train} | {f.n_test} | {f.r2:+.4f} | {_fmt(f.mae)} | {_fmt(f.rmse)} | {_fmt(f.explained_variance)} |"
            )
        lines.append("")

    lines += ["## Target Suitability (Phase 6)", "", "| Target | Verdict | CV mean R2 | Max |Pearson| | Reformulation? |", "|---|:---:|---:|---:|:---:|"]
    for target in NAMED_TARGETS:
        s = suitability[target]
        cv_r2_str = "n/a" if s.cv_mean_r2 is None else f"{s.cv_mean_r2:+.4f}"
        lines.append(f"| {target} | **{s.verdict}** | {cv_r2_str} | {_fmt(s.max_abs_pearson)} | {'yes' if s.suggested_reformulation else 'no'} |")
    lines.append("")
    for target in NAMED_TARGETS:
        s = suitability[target]
        lines += [f"### `{target}` -- {s.verdict}", ""]
        for reason in s.reasons:
            lines.append(f"- {reason}")
        if s.suggested_reformulation:
            lines.append(f"- **Suggested reformulation**: {s.suggested_reformulation}")
        lines.append("")

    lines += [
        "## Generalization Analysis", "",
        "Train R2 vs. cross-validated R2 -- a large positive gap is the textbook overfitting signature.",
        "",
        "**Note on the CV mean R2 column below**: this is a *different, narrower* measurement than the "
        "'After (CV mean R2)' column in the Performance Comparison table above. This one is the trainer's "
        "own internal check (`RegressionConfig.enable_cross_validation=True`), computed only on the "
        "before-split's training portion, after the full production `FeaturePipeline`/`FeatureScaler` "
        "transform. The Performance Comparison table's number is this script's own Phase 4 walk-forward CV, "
        "run across the *entire* dataset on raw (only standard-scaled) features -- more statistical power, "
        "and the primary 'after' evidence for this report. Both are legitimate and expected to differ; "
        "neither is a bug.",
        "",
        "| Target | Train R2 | Holdout R2 | CV mean R2 (train-split only) | Generalization score | Model health score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in NAMED_TARGETS:
        model = improved_pipelines[target].model
        assert model is not None
        h = model.health_scores_
        cv_r2 = "n/a" if model.cv_mean_r2_ is None else f"{model.cv_mean_r2_:+.4f}"
        lines.append(
            f"| {target} | {model.train_r2_:+.4f} | {model.historical_r2_:+.4f} | {cv_r2} | "
            f"{h.generalization_score:.1f} | {h.model_health_score:.1f} |"
        )
    lines.append("")

    lines += [
        "## Feature Diagnostics Summary", "",
        "Full per-target detail (Pearson/Spearman/mutual information/VIF/near-constant/high-correlation "
        "pairs/model-based feature importance) regenerated in `FEATURE_DIAGNOSTICS_REPORT.md`.",
        "",
        "## Residual Analysis Summary", "",
        "Full per-target detail (normality, heteroscedasticity, Durbin-Watson, autocorrelation, "
        "trending/ranging/volatility/session regime breakdown) regenerated in `RESIDUAL_ANALYSIS_REPORT.md`.",
        "",
        "## Decision Engine Metadata (Phase 8)", "",
        "| Target | Model health | Generalization | CV score | Feature quality | Target reliability |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in NAMED_TARGETS:
        h = improved_pipelines[target].model.health_scores_
        lines.append(
            f"| {target} | {h.model_health_score:.1f} | {h.generalization_score:.1f} | "
            f"{h.cross_validation_score:.1f} | {h.feature_quality_score:.1f} | {h.target_reliability:.1f} |"
        )
    lines.append("")

    if live_prediction is not None:
        lines += [
            "### Live prediction (end-to-end Phase 8 evidence)", "",
            "One real `RegressionEngine.predict()` call against a live `MarketState`, using the improved "
            "models above -- confirms Phase 8 metadata reaches an actual prediction, not just the model artifact:",
            "",
            "```json",
        ]
        import json
        lines.append(json.dumps(live_prediction, indent=2, default=str))
        lines += ["```", ""]

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def main() -> None:
    print(f"OANDA environment : {'PRACTICE' if PRACTICE else 'LIVE'} ({BASE_URL})")
    symbol, timeframe = "EUR_USD", "M5"
    window_size, horizon, stride = 250, 5, 3
    candles, dataset, split = build_dataset(symbol, timeframe, 2500, window_size, horizon, stride)

    before = train_baseline(dataset, split, candles, symbol, timeframe, horizon)

    print("\n[Phase 5 + 4] Model selection + walk-forward cross-validation per target...")
    comparisons: Dict[str, List[ModelComparisonResult]] = {}
    best_types: Dict[str, str] = {}
    best_hyperparams: Dict[str, Dict[str, float]] = {}
    cv_results: Dict[str, Optional[CrossValidationResult]] = {}
    for target in NAMED_TARGETS:
        comparison, best_type, hyperparams, cv_result = select_and_validate(dataset, split, target)
        comparisons[target] = comparison
        best_types[target] = best_type
        best_hyperparams[target] = hyperparams
        cv_results[target] = cv_result
        cv_desc = f"CV mean R2={cv_result.mean_r2:+.4f}" if cv_result else "CV skipped"
        print(f"  {target:<28} winner={best_type:<10} {cv_desc}")

    print("\n[Final] Retraining the Phase-5 winner per target through the full production pipeline...")
    improved: Dict[str, ExperimentRecord] = {}
    for target in NAMED_TARGETS:
        improved[target] = train_improved(
            dataset, split, candles, target, best_types[target], best_hyperparams[target],
            symbol, timeframe, horizon,
        )

    print("\n[Phase 1/2/3] Re-running diagnostics against the improved models...")
    target_diag = analyze_all_targets(dataset.y_reg, dataset.X, dataset.feature_names)
    feature_audits: Dict[str, FeatureAuditReport] = {
        target: audit_features_for_target(dataset.X, dataset.y_reg[target], dataset.feature_names, target)
        for target in NAMED_TARGETS
    }
    X_test = dataset.X[split.test_idx]
    feature_names = dataset.feature_names
    improved_pipelines: Dict[str, RegressionInferencePipeline] = {}
    residual_diag: Dict[str, ResidualDiagnostics] = {}
    for target, record in improved.items():
        pipeline = RegressionInferencePipeline(record.artifact_dir, feature_version="1.0.0", strict=True).load()
        improved_pipelines[target] = pipeline
        point, _std = pipeline.predict(X_test, feature_names)
        y_pred = point.ravel()
        y_true = dataset.y_reg[target][split.test_idx]
        residual_diag[target] = analyze_residuals(target, y_true, y_pred, X_test, feature_names)

    print("[Phase 6] Assessing target suitability...")
    suitability: Dict[str, TargetSuitability] = {}
    for target in NAMED_TARGETS:
        b = before[target].testing_metrics
        cv = cv_results.get(target)
        suitability[target] = assess_target_suitability(
            target, r2_before=b["r2"], cv_mean_r2=cv.mean_r2 if cv else None, cv_std_r2=cv.std_r2 if cv else None,
            target_diag=target_diag[target], feature_audit=feature_audits[target], residual_diag=residual_diag[target],
        )
        print(f"  {target:<28} {suitability[target].verdict}")

    print("\n[Phase 8] Live prediction against the improved models...")
    live_prediction: Optional[Dict[str, Any]] = None
    try:
        combined_training_config = TrainingConfig(
            experiment_name="lr_improvement_combined", feature_version="1.0.0", dataset_version="1.0.0",
            strategy_version="1.0.0", output_root=str(OUTPUT_ROOT), scaler="standard", random_seed=42,
        )
        combined_config = RegressionConfig(
            targets=NAMED_TARGETS, prediction_horizon=horizon, model_type="ridge",
            pip_size=0.0001, symbol=symbol, timeframe=timeframe, training_config=combined_training_config,
        )
        combined_engine = RegressionEngine(combined_config)
        combined_engine.load({t: r.artifact_dir for t, r in improved.items()})

        mse_engine = MarketStructureEngine(EngineConfig(swing_window=5))
        mse_engine.load(candles.iloc[-window_size:])
        mse_engine.analyze()
        market_state = mse_engine.market_state()
        prediction = combined_engine.predict(market_state, symbol=symbol, timeframe=timeframe)
        live_prediction = prediction.to_dict()
        print(f"  prediction_confidence={live_prediction['prediction_confidence']:.1f} "
              f"model_health_score={live_prediction['model_health_score']}")
    except Exception as exc:  # noqa: BLE001 -- the report is still valuable without this section
        print(f"  [warn] live prediction demo skipped: {exc}")

    print("\nRegenerating Phase 1-3 reports against the improved models...")
    (ROOT / "TARGET_ANALYSIS_REPORT.md").write_text(diag.render_target_report(target_diag), encoding="utf-8")
    (ROOT / "FEATURE_DIAGNOSTICS_REPORT.md").write_text(diag.render_feature_report(feature_audits), encoding="utf-8")
    (ROOT / "RESIDUAL_ANALYSIS_REPORT.md").write_text(diag.render_residual_report(residual_diag), encoding="utf-8")

    report = render_report(
        before=before, comparisons=comparisons, best_types=best_types, cv_results=cv_results,
        improved=improved, improved_pipelines=improved_pipelines, suitability=suitability,
        live_prediction=live_prediction, n_samples=len(dataset), n_features=dataset.X.shape[1],
    )
    (ROOT / "LINEAR_REGRESSION_IMPROVEMENT_REPORT.md").write_text(report, encoding="utf-8")

    print("\nWrote:")
    print(f"  {ROOT / 'LINEAR_REGRESSION_IMPROVEMENT_REPORT.md'}")
    print(f"  {ROOT / 'TARGET_ANALYSIS_REPORT.md'}")
    print(f"  {ROOT / 'FEATURE_DIAGNOSTICS_REPORT.md'}")
    print(f"  {ROOT / 'RESIDUAL_ANALYSIS_REPORT.md'}")


if __name__ == "__main__":
    main()
