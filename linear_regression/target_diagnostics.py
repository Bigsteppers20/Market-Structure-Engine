"""Phase 1 -- per-target statistical audit (feeds ``TARGET_ANALYSIS_REPORT.md``).

Pure, dependency-light diagnostics over a single regression target's
already-computed values (``ml_pipeline.Dataset.y_reg[target]``) and its
correlation against the already-computed feature matrix
(``ml_pipeline.Dataset.X``). Nothing here trains a model, reads a raw
candle, or recomputes a feature -- it is a read-only statistical summary.

The stationarity test (Augmented Dickey-Fuller) is implemented manually via
OLS (``numpy.linalg.lstsq``) rather than importing ``statsmodels`` --
consistent with this platform's existing convention of computing VIF
manually rather than adding that dependency (see
``scripts/feature_correlation_analysis.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

#: MacKinnon (2010) approximate ADF critical values, constant-no-trend
#: variant -- the standard practical case for a financial target series.
#: Hardcoded rather than looked up from a stats-table library.
ADF_CRITICAL_VALUES: Dict[str, float] = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


@dataclass(slots=True)
class StationarityResult:
    adf_statistic: float
    critical_values: Dict[str, float]
    is_stationary_5pct: bool
    n_lags: int
    n_effective_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adf_statistic": None if not np.isfinite(self.adf_statistic) else round(self.adf_statistic, 4),
            "critical_values": self.critical_values, "is_stationary_5pct": self.is_stationary_5pct,
            "n_lags": self.n_lags, "n_effective_samples": self.n_effective_samples,
        }


def augmented_dickey_fuller(series: np.ndarray, max_lags: Optional[int] = None) -> StationarityResult:
    """Manual ADF test (constant, no trend):

        dy[t] = alpha + rho * y[t] + sum_{k=1}^{p} beta_k * dy[t-k] + e[t]

    The null hypothesis (``rho == 0``, a unit root / non-stationary series)
    is rejected -- i.e. the series is stationary at the 5% level -- when the
    t-statistic on ``rho`` falls below ``ADF_CRITICAL_VALUES["5%"]``.
    """
    y = np.asarray(series, dtype=float)
    n = y.size
    if n < 20:
        return StationarityResult(float("nan"), dict(ADF_CRITICAL_VALUES), False, 0, 0)

    if max_lags is None:
        max_lags = max(1, int(np.floor(12 * (n / 100) ** 0.25)))
    max_lags = min(max_lags, max(1, n // 4))

    dy = np.diff(y)  # dy[t] = y[t+1] - y[t], valid for t in [0, n-2]
    t_index = np.arange(max_lags, n - 1)
    n_eff = t_index.size
    if n_eff < 10:
        return StationarityResult(float("nan"), dict(ADF_CRITICAL_VALUES), False, max_lags, n_eff)

    y_dep = dy[t_index]
    y_level = y[t_index]
    lag_cols = [dy[t_index - k] for k in range(1, max_lags + 1)]
    X_design = np.column_stack([np.ones(n_eff), y_level, *lag_cols])

    coeffs, *_ = np.linalg.lstsq(X_design, y_dep, rcond=None)
    residuals = y_dep - X_design @ coeffs
    dof = n_eff - X_design.shape[1]
    if dof <= 0:
        return StationarityResult(float("nan"), dict(ADF_CRITICAL_VALUES), False, max_lags, n_eff)

    sigma2 = float(np.sum(residuals ** 2) / dof)
    xtx_inv = np.linalg.pinv(X_design.T @ X_design)
    se_rho = float(np.sqrt(sigma2 * xtx_inv[1, 1])) if sigma2 >= 0 else float("nan")
    rho = float(coeffs[1])
    t_stat = rho / se_rho if (se_rho == se_rho and se_rho > 0) else float("nan")
    is_stationary = (t_stat == t_stat) and (t_stat < ADF_CRITICAL_VALUES["5%"])
    return StationarityResult(t_stat, dict(ADF_CRITICAL_VALUES), bool(is_stationary), max_lags, n_eff)


def _outlier_percentage(y: np.ndarray, z_threshold: float = 3.0) -> float:
    std = float(y.std())
    if std < 1e-15:
        return 0.0
    z = np.abs((y - y.mean()) / std)
    return float(np.mean(z > z_threshold) * 100.0)


@dataclass(slots=True)
class TargetDiagnostics:
    """Complete Phase 1 audit for one regression target."""

    name: str
    n_samples: int
    mean: float
    std: float
    variance: float
    minimum: float
    maximum: float
    median: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float
    outlier_percentage: float
    missing_percentage: float
    stationarity: StationarityResult
    top_correlated_features: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "n_samples": self.n_samples, "mean": self.mean, "std": self.std,
            "variance": self.variance, "min": self.minimum, "max": self.maximum, "median": self.median,
            "q25": self.q25, "q75": self.q75, "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4), "outlier_percentage": round(self.outlier_percentage, 2),
            "missing_percentage": round(self.missing_percentage, 2), "stationarity": self.stationarity.to_dict(),
            "top_correlated_features": [(n, round(r, 4)) for n, r in self.top_correlated_features],
        }


def analyze_target(
    name: str, y: np.ndarray, X: np.ndarray, feature_names: Sequence[str], top_n: int = 15,
) -> TargetDiagnostics:
    """Full Phase 1 audit for one target's already-computed value array."""
    y_raw = np.asarray(y, dtype=float)
    missing_mask = ~np.isfinite(y_raw)
    missing_percentage = float(np.mean(missing_mask) * 100.0)
    y_clean = y_raw[~missing_mask]
    X_clean = X[~missing_mask] if missing_mask.any() else X

    if y_clean.size == 0:
        raise ValueError(f"Target {name!r} has no finite values to analyze (100% missing).")

    correlations: List[Tuple[str, float]] = []
    y_std = float(y_clean.std())
    for j, fname in enumerate(feature_names):
        col = X_clean[:, j]
        col_std = float(col.std())
        if col_std < 1e-15 or y_std < 1e-15:
            r = 0.0
        else:
            r = float(np.corrcoef(col, y_clean)[0, 1])
            r = 0.0 if not np.isfinite(r) else r
        correlations.append((fname, r))
    correlations.sort(key=lambda kv: abs(kv[1]), reverse=True)

    return TargetDiagnostics(
        name=name, n_samples=int(y_clean.size), mean=float(y_clean.mean()), std=y_std,
        variance=float(y_clean.var()), minimum=float(y_clean.min()), maximum=float(y_clean.max()),
        median=float(np.median(y_clean)), q25=float(np.percentile(y_clean, 25)),
        q75=float(np.percentile(y_clean, 75)), skewness=float(stats.skew(y_clean)),
        kurtosis=float(stats.kurtosis(y_clean)), outlier_percentage=_outlier_percentage(y_clean),
        missing_percentage=missing_percentage, stationarity=augmented_dickey_fuller(y_clean),
        top_correlated_features=correlations[:top_n],
    )


def analyze_all_targets(
    y_reg: Dict[str, np.ndarray], X: np.ndarray, feature_names: Sequence[str], top_n: int = 15,
) -> Dict[str, TargetDiagnostics]:
    return {name: analyze_target(name, y, X, feature_names, top_n=top_n) for name, y in y_reg.items()}
