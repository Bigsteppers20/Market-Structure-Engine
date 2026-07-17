"""End-to-end ml_pipeline demonstration against real OANDA historical data.

Fetches real (never mock) EUR/USD M5 candles, builds a leakage-safe rolling-
window dataset via :class:`ml_pipeline.DatasetBuilder`, splits it
chronologically, runs the feature pipeline / scaler / an example feature
selector, exports the result in every supported format, and writes
``DATASET_REPORT.md`` at the project root summarizing this exact run.

No model is trained here -- this script's output is the dataset, ready for
a separate training step.

Run from the project root:

    .venv\\Scripts\\python.exe examples\\build_dataset_example.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ml_pipeline import (
    DatasetConfig,
    DatasetBuilder,
    DatasetExporter,
    FeaturePipeline,
    FeatureScaler,
    FeatureSelector,
    TimeSeriesSplitter,
)
from oanda_client import BASE_URL, PRACTICE, fetch_candles

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "dataset_output"


def main() -> None:
    print(f"OANDA environment : {'PRACTICE' if PRACTICE else 'LIVE'} ({BASE_URL})")
    symbol, timeframe = "EUR_USD", "M5"
    raw = fetch_candles(symbol, timeframe, count=3000)
    print(f"Fetched {len(raw)} live {symbol} {timeframe} candles "
          f"({raw['timestamp'].iloc[0]} -> {raw['timestamp'].iloc[-1]})")

    cfg = DatasetConfig(
        symbol=symbol,
        timeframe=timeframe,
        window_size=300,
        horizon=3,
        stride=3,
        regression_targets=[
            "next_close", "next_return", "next_log_return",
            "expected_pip_movement", "future_atr", "future_volatility",
        ],
        classification_label="threshold",
        classification_kwargs={"buy_threshold": 0.0006, "sell_threshold": -0.0006},
        impute_invalid=True,
        scaler="standard",
    )

    print(f"\nBuilding dataset: window_size={cfg.window_size}, horizon={cfg.horizon}, "
          f"stride={cfg.stride} (one MarketStructureEngine.analyze() call per sample)...")
    t0 = time.time()
    builder = DatasetBuilder(cfg)
    dataset = builder.build(raw)
    elapsed = time.time() - t0
    report = builder.last_report
    print(f"Built {len(dataset)} samples x {dataset.X.shape[1]} features in {elapsed:.1f}s")
    print(f"Input validation issues: {builder.last_input_issues or 'none'}")
    print(f"Dataset valid: {report.is_valid} | leakage_ok: {report.leakage_ok}")

    # --- chronological split -------------------------------------------- #
    splitter = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15)
    split = splitter.split(len(dataset))[0]
    print(f"\nSplit -> train={len(split.train_idx)} val={len(split.val_idx)} test={len(split.test_idx)}")

    # --- feature pipeline (fit on train only) ---------------------------- #
    fp = FeaturePipeline(cfg)
    X_train, feat_names = fp.fit_transform(dataset.X[split.train_idx], dataset.feature_names)
    X_val, _ = fp.transform(dataset.X[split.val_idx], dataset.feature_names)
    X_test, _ = fp.transform(dataset.X[split.test_idx], dataset.feature_names)

    # --- scaling (fit on train only) ------------------------------------- #
    scaler = FeatureScaler(cfg.scaler)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    scaler_path = OUTPUT_DIR / "feature_scaler.joblib"
    OUTPUT_DIR.mkdir(exist_ok=True)
    scaler.save(scaler_path)
    print(f"Scaler ({cfg.scaler}) fit on train, saved to {scaler_path}")
    print(f"  -> transformed shapes: train={X_train_scaled.shape} "
          f"val={X_val_scaled.shape} test={X_test_scaled.shape} (same scaler, no refitting)")

    # --- optional feature selection demo (correlation filter) ------------ #
    selector = FeatureSelector("correlation", correlation_threshold=0.95)
    y_train_cls = dataset.y_cls[split.train_idx]
    x_train_selected, selected_names = selector.fit_transform(X_train_scaled, y_train_cls, feat_names)
    print(f"FeatureSelector(correlation, threshold=0.95) kept {x_train_selected.shape[1]} / "
          f"{len(feat_names)} features on the train split (demonstration only -- "
          f"exported dataset below keeps all {len(feat_names)}).")

    # --- export the full (unselected) dataset in every format ------------ #
    for fmt, fn in (
        ("csv", DatasetExporter.to_csv), ("parquet", DatasetExporter.to_parquet),
        ("npz", DatasetExporter.to_numpy), ("joblib", DatasetExporter.to_joblib),
        ("pkl", DatasetExporter.to_pickle),
    ):
        path = fn(dataset, OUTPUT_DIR / f"dataset.{fmt}")
        print(f"Exported {fmt:<8} -> {path} ({path.stat().st_size:,} bytes)")

    # --- write DATASET_REPORT.md from this exact run --------------------- #
    _write_report(
        cfg=cfg, dataset=dataset, report=report, split=split,
        feat_names=feat_names, selected_names=selected_names,
        X_train_scaled=X_train_scaled, raw=raw,
        input_issues=builder.last_input_issues, elapsed=elapsed,
    )
    print(f"\nWrote {ROOT / 'DATASET_REPORT.md'}")


def _write_report(*, cfg, dataset, report, split, feat_names, selected_names,
                   X_train_scaled, raw, input_issues, elapsed) -> None:
    n = len(dataset)
    lines: list[str] = []
    a = lines.append

    a("# Dataset Report")
    a("")
    a("Generated by `examples/build_dataset_example.py` against **real OANDA historical "
      "data** (not synthetic) -- every number below comes from that actual run.")
    a("")
    a("## Source Data")
    a("")
    a(f"- Symbol / timeframe: `{cfg.symbol}` / `{cfg.timeframe}`")
    a(f"- Raw candles fetched: {len(raw)} "
      f"({raw['timestamp'].iloc[0]} -> {raw['timestamp'].iloc[-1]})")
    a(f"- Input validation issues: {input_issues or 'none'}")
    a("")
    a("## Build Configuration")
    a("")
    a(f"- window_size: {cfg.window_size} candles per feature-vector sample")
    a(f"- horizon: {cfg.horizon} bars ahead for every label")
    a(f"- stride: {cfg.stride} (1 sample every {cfg.stride} candles)")
    a(f"- regression_targets: {cfg.regression_targets}")
    a(f"- classification_label: {cfg.classification_label} "
      f"(kwargs={cfg.classification_kwargs})")
    a(f"- Build time: {elapsed:.1f}s for {n} `MarketStructureEngine.analyze()` calls "
      f"({elapsed / max(n, 1) * 1000:.1f} ms/sample)")
    a("")
    a("## Number of Samples")
    a("")
    a(f"- Total samples: **{n}**")
    a(f"- Train: {len(split.train_idx)} | Validation: {len(split.val_idx)} | Test: {len(split.test_idx)}")
    a("")
    a("## Number of Features")
    a("")
    a(f"- Raw engine feature count: **{dataset.X.shape[1]}** (unchanged from the "
      "Market Structure Engine's own `feature_vector()` -- this package never modifies it)")
    a(f"- After FeaturePipeline (imputation/encoding, no selection): {len(feat_names)}")
    a(f"- After demonstration `FeatureSelector(\"correlation\", threshold=0.95)` on train: "
      f"{len(selected_names)} (informational only -- the exported dataset keeps all "
      f"{len(feat_names)} columns; selection is an opt-in step for a specific model)")
    a("")
    a("## Missing Values")
    a("")
    a(f"- NaN count in final X: {report.nan_count}")
    a(f"- Inf count in final X: {report.inf_count}")
    a("- Strategy: every gated engine feature carries a paired `_valid` flag (never a "
      "silent 0.0 -- see the Market Structure Engine's FEATURE_OPTIMIZATION_REPORT.md); "
      f"`FeaturePipeline(impute_invalid={cfg.impute_invalid})` replaces invalid-flagged "
      "values with the **training-split median** before scaling.")
    a("")
    a("## Scaling Summary")
    a("")
    per_col_std = X_train_scaled.std(axis=0)
    n_near_constant = int((per_col_std < 1e-6).sum())
    a(f"- Method: `{cfg.scaler}` (scikit-learn `StandardScaler`), fit on the {len(split.train_idx)}-sample "
      "train split only, reused unchanged for validation/test/inference")
    a(f"- Post-scaling train-split global mean (should be ~0): {float(np.mean(X_train_scaled)):.6f}")
    a(f"- Post-scaling per-feature std: median {float(np.median(per_col_std)):.4f}, "
      f"mean {float(np.mean(per_col_std)):.4f} (each *non-constant* column is individually "
      "scaled to std=1; the mean is pulled below 1 by columns that are genuinely constant "
      f"within this window_size, e.g. `_valid` flags already past warm-up -- "
      f"{n_near_constant} of {per_col_std.size} columns have std < 1e-6 here)")
    a("- Scaler persisted to: `dataset_output/feature_scaler.joblib`")
    a("")
    a("## Leakage Checks")
    a("")
    a(f"- `leakage_ok`: **{report.leakage_ok}** -- every one of the {n} samples passed "
      "`assert_no_future_leakage()` during construction (window end == decision bar, "
      "label strictly after it); the build would have raised `LeakageError` otherwise.")
    a("- Feature window never overlaps the label horizon by construction (see "
      "`dataset_builder.py::DatasetBuilder.build`).")
    a("")
    a("## Class Balance (classification labels)")
    a("")
    a("| Class | Count | Fraction |")
    a("|---|---|---|")
    for cls_name, count in report.class_balance.items():
        a(f"| {cls_name} | {count} | {count / n:.1%} |")
    a("")
    a("## Regression Target Distribution")
    a("")
    a("| Target | Mean | Std | Min | Median | Max |")
    a("|---|---|---|---|---|---|")
    for name, stats in report.regression_stats.items():
        a(f"| `{name}` | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | "
          f"{stats['median']:.6f} | {stats['max']:.6f} |")
    a("")
    a("## Train / Validation / Test Sizes")
    a("")
    a(f"- Train: {len(split.train_idx)} samples "
      f"({dataset.metadata['timestamp'].iloc[split.train_idx[0]]} -> "
      f"{dataset.metadata['timestamp'].iloc[split.train_idx[-1]]})")
    a(f"- Validation: {len(split.val_idx)} samples "
      f"({dataset.metadata['timestamp'].iloc[split.val_idx[0]]} -> "
      f"{dataset.metadata['timestamp'].iloc[split.val_idx[-1]]})")
    a(f"- Test: {len(split.test_idx)} samples "
      f"({dataset.metadata['timestamp'].iloc[split.test_idx[0]]} -> "
      f"{dataset.metadata['timestamp'].iloc[split.test_idx[-1]]})")
    a("- Split is strictly chronological (train entirely precedes validation, which "
      "entirely precedes test) -- enforced by `SplitResult.__post_init__`, never shuffled.")
    a("")
    a("## Validation Summary")
    a("")
    a(f"- `report.is_valid`: **{report.is_valid}**")
    a(f"- Feature count OK: {report.feature_count_ok}")
    a(f"- Target alignment OK: {report.target_alignment_ok}")
    a(f"- Duplicate metadata rows: {report.duplicate_metadata_rows}")
    a(f"- Errors: {report.errors or 'none'}")
    a(f"- Warnings: {report.warnings or 'none'}")
    a("")

    (ROOT / "DATASET_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
