"""Phase 2 -- feature-level diagnostics (recommendations only; nothing is
ever auto-removed).

Pearson/Spearman-vs-target, Variance Inflation Factor, and near-constant/
high-correlation-pair detection reuse the exact same manual-OLS
methodology already established in
``scripts/feature_correlation_analysis.py`` (Task 9 of
``FEATURE_OPTIMIZATION_REPORT.md``) -- independently implemented here
(not imported from that one-off script) to keep this engine self-contained,
the same "consistent methodology, independent per layer" convention this
platform already uses for ``linear_regression``/``logistic_regression``'s
otherwise-identical ``feature_mapper.py`` modules. Mutual Information and
feature importance are genuinely new here (target-aware, which the
existing script is not).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from .regression_model import RegressionModel

HIGH_CORR_THRESHOLD = 0.95
HIGH_VIF_THRESHOLD = 10.0
LOW_VARIANCE_THRESHOLD = 1e-8
IMBALANCE_THRESHOLD = 0.99


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-15 or b.std() < 1e-15:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    return r if np.isfinite(r) else 0.0


def pearson_vs_target(X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]) -> Dict[str, float]:
    return {name: _pearson(X[:, j], y) for j, name in enumerate(feature_names)}


def spearman_vs_target(X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]) -> Dict[str, float]:
    y_rank = pd.Series(y).rank().to_numpy()
    out = {}
    for j, name in enumerate(feature_names):
        x_rank = pd.Series(X[:, j]).rank().to_numpy()
        out[name] = _pearson(x_rank, y_rank)
    return out


def mutual_information_vs_target(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str], random_state: int = 42,
) -> Dict[str, float]:
    """Mutual information (nats) between each feature and the target --
    captures non-linear dependence Pearson/Spearman cannot, which is
    exactly the gap a *linear* model's correlation-based diagnostics miss."""
    mi = mutual_info_regression(X, y, random_state=random_state)
    return {name: float(v) for name, v in zip(feature_names, mi)}


def feature_importance_vs_target(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
    model_type: str = "ridge", alpha: float = 1.0, random_state: int = 42,
) -> Dict[str, float]:
    """Model-based feature importance (mean |coefficient|) for this specific
    target -- fits a throwaway, unregularized-by-default ridge (bootstrap
    disabled) via the same ``RegressionModel`` the trainer uses, then reuses
    its existing ``feature_importance()`` (never reimplements the coefficient
    extraction). Ridge by default rather than plain linear regression: with
    ~185 correlated features a plain OLS fit is exactly the unstable,
    small-data-change-swings-coefficients regime the VIF diagnostics above
    warn about, which would make the resulting importances themselves noisy."""
    model = RegressionModel(model_type=model_type, n_bootstrap=0, random_state=random_state, alpha=alpha)
    model.fit(X, y, ["_importance_probe"])
    return model.feature_importance(list(feature_names)) or {}


def variance_inflation_factors(X: np.ndarray, feature_names: Sequence[str]) -> Dict[str, float]:
    """Manual VIF via OLS R^2 (no statsmodels dependency -- same convention
    as ``scripts/feature_correlation_analysis.py``'s ``vif_report()``)."""
    keep_mask = X.std(axis=0, ddof=0) > 1e-10
    kept_idx = np.nonzero(keep_mask)[0]
    Xk = X[:, kept_idx]
    std = Xk.std(axis=0, ddof=0)
    Xk = (Xk - Xk.mean(axis=0)) / np.where(std == 0, 1.0, std)

    vifs = np.full(X.shape[1], np.nan)
    n, p = Xk.shape
    for local_j in range(p):
        target_col = Xk[:, local_j]
        others = np.delete(Xk, local_j, axis=1)
        design = np.column_stack([np.ones(n), others])
        coef, *_ = np.linalg.lstsq(design, target_col, rcond=None)
        pred = design @ coef
        ss_res = float(np.sum((target_col - pred) ** 2))
        ss_tot = float(np.sum((target_col - target_col.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r2 = min(max(r2, 0.0), 1 - 1e-9)
        vifs[kept_idx[local_j]] = 1.0 / (1.0 - r2)
    return {name: float(v) for name, v in zip(feature_names, vifs)}


def near_constant_features(X: np.ndarray, feature_names: Sequence[str]) -> List[str]:
    variance = X.var(axis=0, ddof=0)
    flagged = []
    for j, name in enumerate(feature_names):
        col = X[:, j]
        is_binary = set(np.unique(col)) <= {0.0, 1.0}
        majority_frac = float(pd.Series(col).value_counts(normalize=True).iloc[0])
        if variance[j] < LOW_VARIANCE_THRESHOLD or (is_binary and majority_frac >= IMBALANCE_THRESHOLD):
            flagged.append(name)
    return flagged


def highly_correlated_pairs(X: np.ndarray, feature_names: Sequence[str]) -> List[Tuple[str, str, float]]:
    df = pd.DataFrame(X, columns=feature_names)
    keep = df.columns[df.std(axis=0, ddof=0) > 1e-10]
    pearson = df[keep].corr(method="pearson")
    cols = list(pearson.columns)
    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = pearson.iloc[i, j]
            if pd.notna(r) and abs(r) >= HIGH_CORR_THRESHOLD:
                pairs.append((cols[i], cols[j], float(r)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs


@dataclass(slots=True)
class FeatureAuditReport:
    """Complete Phase 2 audit for one target."""

    target: str
    pearson: Dict[str, float]
    spearman: Dict[str, float]
    mutual_information: Dict[str, float]
    vif: Dict[str, float]
    near_constant: List[str]
    high_correlation_pairs: List[Tuple[str, str, float]]
    recommendations: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "pearson": {k: round(v, 4) for k, v in self.pearson.items()},
            "spearman": {k: round(v, 4) for k, v in self.spearman.items()},
            "mutual_information": {k: round(v, 4) for k, v in self.mutual_information.items()},
            "vif": {k: (None if not np.isfinite(v) else round(v, 2)) for k, v in self.vif.items()},
            "near_constant": self.near_constant,
            "high_correlation_pairs": [(a, b, round(r, 4)) for a, b, r in self.high_correlation_pairs],
            "recommendations": self.recommendations,
            "feature_importance": {k: round(v, 4) for k, v in self.feature_importance.items()},
        }


def _generate_recommendations(
    *, pearson: Dict[str, float], mutual_information: Dict[str, float], vif: Dict[str, float],
    near_constant: List[str], high_correlation_pairs: List[Tuple[str, str, float]],
    mi_low_threshold: float = 0.01,
) -> List[str]:
    """Textual, human-reviewable recommendations only -- nothing here
    removes a feature. Every recommendation names the concrete evidence
    (not a bare "low value") so a reviewer can independently verify it."""
    recs: List[str] = []

    if near_constant:
        recs.append(
            f"{len(near_constant)} near-constant feature(s) carry almost no information for ANY target "
            f"(same list regardless of target): {', '.join(near_constant[:10])}"
            + (", ..." if len(near_constant) > 10 else "") + ". Candidates for removal only if this holds across ALL targets, not just this one."
        )

    for a, b, r in high_correlation_pairs[:15]:
        mi_a, mi_b = mutual_information.get(a, 0.0), mutual_information.get(b, 0.0)
        weaker = a if mi_a < mi_b else b
        stronger = b if weaker == a else a
        recs.append(
            f"'{a}' and '{b}' are highly correlated (Pearson r={r:+.3f}) -- carry near-duplicate information. "
            f"'{weaker}' has lower mutual information with this target ({mi_a if weaker==a else mi_b:.4f} vs "
            f"{mi_b if weaker==a else mi_a:.4f} for '{stronger}') -- if only one is kept, prefer '{stronger}'."
        )

    high_vif = sorted(
        ((name, v) for name, v in vif.items() if np.isfinite(v) and v > HIGH_VIF_THRESHOLD),
        key=lambda kv: kv[1], reverse=True,
    )
    if high_vif:
        names = ", ".join(f"{n} (VIF={v:.1f})" for n, v in high_vif[:10])
        recs.append(
            f"{len(high_vif)} feature(s) have VIF > {HIGH_VIF_THRESHOLD} (severe multicollinearity): {names}"
            + (", ..." if len(high_vif) > 10 else "") + ". A linear model's coefficients on these are unstable "
            "(small data changes swing them sharply) even if overall R^2 looks fine -- consider regularization "
            "(ridge/lasso, already supported) over removal, since removal changes what the coefficients mean."
        )

    low_info = [
        name for name in pearson
        if abs(pearson[name]) < 0.02 and mutual_information.get(name, 0.0) < mi_low_threshold
        and name not in near_constant
    ]
    if low_info:
        recs.append(
            f"{len(low_info)} feature(s) show both near-zero linear correlation (|Pearson| < 0.02) and near-zero "
            f"mutual information (< {mi_low_threshold}) with this specific target -- likely low-information FOR "
            f"THIS TARGET specifically (may still matter for other targets or for the Logistic Regression Engine): "
            + ", ".join(low_info[:10]) + (", ..." if len(low_info) > 10 else "") + "."
        )

    if not recs:
        recs.append("No high-severity feature issues detected for this target at the configured thresholds.")
    return recs


def audit_features_for_target(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str], target_name: str,
) -> FeatureAuditReport:
    pearson = pearson_vs_target(X, y, feature_names)
    spearman = spearman_vs_target(X, y, feature_names)
    mi = mutual_information_vs_target(X, y, feature_names)
    vif = variance_inflation_factors(X, feature_names)
    near_constant = near_constant_features(X, feature_names)
    high_corr_pairs = highly_correlated_pairs(X, feature_names)
    importance = feature_importance_vs_target(X, y, feature_names)
    recommendations = _generate_recommendations(
        pearson=pearson, mutual_information=mi, vif=vif, near_constant=near_constant,
        high_correlation_pairs=high_corr_pairs,
    )
    return FeatureAuditReport(
        target=target_name, pearson=pearson, spearman=spearman, mutual_information=mi, vif=vif,
        near_constant=near_constant, high_correlation_pairs=high_corr_pairs, recommendations=recommendations,
        feature_importance=importance,
    )
