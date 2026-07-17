"""Feature correlation / multicollinearity / variance analysis (Task 9).

Computes Pearson correlation, Spearman correlation, Variance Inflation
Factor (VIF), and a variance-threshold scan over the Market Structure
Engine's feature vector, sampled across real historical candles.

This script does not train or evaluate any ML model -- it only prepares
statistics that inform feature selection for a downstream linear/logistic
regression pipeline (per FEATURE_OPTIMIZATION_REPORT.md, Task 9).

Multiple feature-vector snapshots are required to compute correlation/VIF/
variance (a single `analyze()` call only yields one row). To stay leakage
-safe (see FEATURE_REFERENCE.md, Data-Leakage Analysis), each sample
re-runs the engine on a growing, freshly-sliced prefix of the historical
data (`df.iloc[:i+1]`) rather than reusing full-series internal arrays
across samples.

Usage (from the project root, with a configured .env -- see
examples/live_oanda_test.py for credential setup)::

    .venv\\Scripts\\python.exe scripts\\feature_correlation_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from market_structure import EngineConfig, MarketStructureEngine  # noqa: E402
from oanda_client import BASE_URL, PRACTICE, fetch_candles  # noqa: E402

HIGH_CORR_THRESHOLD = 0.95
HIGH_VIF_THRESHOLD = 10.0
LOW_VARIANCE_THRESHOLD = 1e-8
IMBALANCE_THRESHOLD = 0.99  # binary feature >=99% one class => near-constant


def build_sample_matrix(
    df: pd.DataFrame, warmup: int, step: int, config: EngineConfig | None = None
) -> tuple[np.ndarray, list[str]]:
    """Re-run the engine on growing, leakage-safe prefixes of `df` and collect vectors."""
    engine = MarketStructureEngine(config or EngineConfig(swing_window=5))
    rows: list[np.ndarray] = []
    names: list[str] | None = None
    for i in range(warmup, len(df), step):
        sliced = df.iloc[: i + 1]
        engine.load(sliced)
        engine.analyze()
        vec, vec_names = engine.feature_vector()
        if names is None:
            names = vec_names
        rows.append(vec)
    assert names is not None
    return np.vstack(rows), names


def variance_threshold_report(X: np.ndarray, names: list[str]) -> pd.DataFrame:
    var = X.var(axis=0, ddof=0)
    is_binary = np.array([set(np.unique(X[:, j])) <= {0.0, 1.0} for j in range(X.shape[1])])
    frac_majority = np.array([
        (X[:, j] == pd.Series(X[:, j]).mode().iloc[0]).mean() for j in range(X.shape[1])
    ])
    low_var = (var < LOW_VARIANCE_THRESHOLD) | (is_binary & (frac_majority >= IMBALANCE_THRESHOLD))
    return pd.DataFrame({
        "feature": names, "variance": var, "is_binary": is_binary,
        "majority_class_fraction": np.where(is_binary, frac_majority, np.nan),
        "flag_low_variance": low_var,
    })


def vif_report(X: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Manual VIF via OLS R^2 (no statsmodels dependency, consistent with the
    engine's own 'no external stats/TA library' convention)."""
    keep_mask = X.std(axis=0, ddof=0) > 1e-10  # drop constant columns first
    kept_idx = np.nonzero(keep_mask)[0]
    Xk = X[:, kept_idx]
    Xk = (Xk - Xk.mean(axis=0)) / np.where(Xk.std(axis=0, ddof=0) == 0, 1.0, Xk.std(axis=0, ddof=0))

    vifs = np.full(X.shape[1], np.nan)
    n, p = Xk.shape
    for local_j in range(p):
        y = Xk[:, local_j]
        others = np.delete(Xk, local_j, axis=1)
        design = np.column_stack([np.ones(n), others])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r2 = min(max(r2, 0.0), 1 - 1e-9)
        vifs[kept_idx[local_j]] = 1.0 / (1.0 - r2)
    return pd.DataFrame({"feature": names, "vif": vifs})


def correlation_report(X: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(X, columns=names)
    keep = df.columns[df.std(axis=0, ddof=0) > 1e-10]
    pearson = df[keep].corr(method="pearson")
    spearman = df[keep].corr(method="spearman")

    pairs = []
    cols = list(pearson.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            p = pearson.iloc[i, j]
            if pd.notna(p) and abs(p) >= HIGH_CORR_THRESHOLD:
                pairs.append((cols[i], cols[j], float(p), float(spearman.iloc[i, j])))
    high_corr = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "pearson", "spearman"])
    high_corr = high_corr.reindex(high_corr["pearson"].abs().sort_values(ascending=False).index)
    return pearson, spearman, high_corr


def main() -> None:
    print(f"OANDA environment : {'PRACTICE' if PRACTICE else 'LIVE'} ({BASE_URL})")
    df = fetch_candles("EUR_USD", "M5", 2500)
    print(f"Fetched {len(df)} live EUR_USD M5 candles "
          f"({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")

    warmup, step = 300, 5
    X, names = build_sample_matrix(df, warmup=warmup, step=step)
    print(f"Built sample matrix: {X.shape[0]} snapshots x {X.shape[1]} features "
          f"(warmup={warmup} bars, step={step} bars, leakage-safe re-slice per sample)")

    var_df = variance_threshold_report(X, names)
    vif_df = vif_report(X, names)
    pearson, spearman, high_corr = correlation_report(X, names)

    out_dir = ROOT / "analysis_output"
    out_dir.mkdir(exist_ok=True)
    var_df.to_csv(out_dir / "variance_report.csv", index=False)
    vif_df.to_csv(out_dir / "vif_report.csv", index=False)
    high_corr.to_csv(out_dir / "high_correlation_pairs.csv", index=False)
    pearson.to_csv(out_dir / "pearson_correlation_matrix.csv")
    spearman.to_csv(out_dir / "spearman_correlation_matrix.csv")

    low_var = var_df[var_df["flag_low_variance"]]
    high_vif = vif_df[vif_df["vif"] > HIGH_VIF_THRESHOLD].sort_values("vif", ascending=False)

    print()
    print(f"Low-variance / near-constant features: {len(low_var)}")
    for _, r in low_var.iterrows():
        print(f"  {r['feature']:<35} var={r['variance']:.2e} "
              f"binary={r['is_binary']} majority_frac={r['majority_class_fraction']}")

    print()
    print(f"High-VIF features (VIF > {HIGH_VIF_THRESHOLD}): {len(high_vif)}")
    for _, r in high_vif.head(30).iterrows():
        vif_str = "inf" if not np.isfinite(r["vif"]) else f"{r['vif']:.1f}"
        print(f"  {r['feature']:<35} VIF={vif_str}")

    print()
    print(f"Highly correlated pairs (|Pearson| >= {HIGH_CORR_THRESHOLD}): {len(high_corr)}")
    for _, r in high_corr.head(40).iterrows():
        print(f"  {r['feature_a']:<32} <-> {r['feature_b']:<32} "
              f"pearson={r['pearson']:+.3f} spearman={r['spearman']:+.3f}")

    print()
    print(f"Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
