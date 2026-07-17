"""Phase 3 -- residual analysis (feeds ``RESIDUAL_ANALYSIS_REPORT.md``).

Reuses ``metrics.residual_statistics``/``metrics.prediction_error_distribution``
unmodified (no duplication of distribution/histogram logic already computed
elsewhere in this engine). Normality, heteroscedasticity, autocorrelation,
and the Durbin-Watson statistic are implemented manually via scipy/numpy --
same "no statsmodels" convention as ``target_diagnostics.py``'s ADF test
and ``feature_diagnostics.py``'s VIF.

Regime breakdown (trending/ranging/high-vol/low-vol/session) reads regime
labels directly off already-computed ``MarketState`` feature columns in the
dataset's own feature matrix -- never a raw candle or a recomputed
indicator, and independently implemented here (not imported from
``model_monitor``) to keep this engine decoupled from that peer system,
the same convention already applied between ``linear_regression`` and
``logistic_regression``'s otherwise-identical ``feature_mapper.py`` modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy import stats

from .metrics import prediction_error_distribution, residual_statistics


@dataclass(slots=True)
class QQAnalysis:
    qq_correlation: float
    """Correlation between theoretical normal quantiles and empirical
    residual quantiles -- 1.0 = perfectly normal-shaped tails."""
    n_points: int

    def to_dict(self) -> Dict[str, Any]:
        return {"qq_correlation": round(self.qq_correlation, 4), "n_points": self.n_points}


def qq_analysis(residuals: np.ndarray) -> QQAnalysis:
    e = np.sort(np.asarray(residuals, dtype=float))
    n = e.size
    if n < 3 or e.std() < 1e-15:
        return QQAnalysis(qq_correlation=float("nan"), n_points=n)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(probs, loc=e.mean(), scale=e.std())
    r = float(np.corrcoef(theoretical, e)[0, 1])
    return QQAnalysis(qq_correlation=r if np.isfinite(r) else float("nan"), n_points=n)


@dataclass(slots=True)
class NormalityResult:
    shapiro_statistic: float
    shapiro_pvalue: float
    jarque_bera_statistic: float
    jarque_bera_pvalue: float
    is_normal_5pct: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shapiro_statistic": round(self.shapiro_statistic, 4), "shapiro_pvalue": round(self.shapiro_pvalue, 4),
            "jarque_bera_statistic": round(self.jarque_bera_statistic, 4),
            "jarque_bera_pvalue": round(self.jarque_bera_pvalue, 4), "is_normal_5pct": self.is_normal_5pct,
        }


def test_normality(residuals: np.ndarray, max_shapiro_n: int = 5000, random_state: int = 42) -> NormalityResult:
    r = np.asarray(residuals, dtype=float)
    if r.size > max_shapiro_n:
        rng = np.random.default_rng(random_state)
        sample = rng.choice(r, size=max_shapiro_n, replace=False)
    else:
        sample = r
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    jb_stat, jb_p = stats.jarque_bera(r)
    is_normal = bool(shapiro_p > 0.05 and jb_p > 0.05)
    return NormalityResult(float(shapiro_stat), float(shapiro_p), float(jb_stat), float(jb_p), is_normal)


@dataclass(slots=True)
class HeteroscedasticityResult:
    lm_statistic: float
    lm_pvalue: float
    is_heteroscedastic_5pct: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lm_statistic": round(self.lm_statistic, 4), "lm_pvalue": round(self.lm_pvalue, 4),
            "is_heteroscedastic_5pct": self.is_heteroscedastic_5pct,
        }


def breusch_pagan_test(residuals: np.ndarray, fitted_values: np.ndarray) -> HeteroscedasticityResult:
    """Manual Breusch-Pagan test (squared residuals regressed on fitted
    values): LM statistic = n * R^2 of the auxiliary regression, ~ chi2(1)
    under the null of homoscedasticity."""
    e2 = np.asarray(residuals, dtype=float) ** 2
    f = np.asarray(fitted_values, dtype=float)
    n = f.size
    design = np.column_stack([np.ones(n), f])
    coef, *_ = np.linalg.lstsq(design, e2, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((e2 - pred) ** 2))
    ss_tot = float(np.sum((e2 - e2.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    r2 = min(max(r2, 0.0), 1 - 1e-12)
    lm_stat = n * r2
    p_value = float(1 - stats.chi2.cdf(lm_stat, df=1))
    return HeteroscedasticityResult(lm_stat, p_value, bool(p_value < 0.05))


def durbin_watson(residuals: np.ndarray) -> float:
    """~2.0 = no autocorrelation, <2 = positive autocorrelation, >2 = negative."""
    e = np.asarray(residuals, dtype=float)
    denom = float(np.sum(e ** 2))
    if denom < 1e-15:
        return float("nan")
    return float(np.sum(np.diff(e) ** 2) / denom)


def autocorrelation(residuals: np.ndarray, max_lag: int = 10) -> Dict[int, float]:
    e = np.asarray(residuals, dtype=float)
    e = e - e.mean()
    var = float(np.sum(e ** 2))
    out: Dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        if lag >= e.size:
            break
        out[lag] = float(np.sum(e[:-lag] * e[lag:]) / var) if var > 0 else 0.0
    return out


def _regime_masks(X: np.ndarray, feature_names: Sequence[str]) -> Dict[str, np.ndarray]:
    idx = {name: i for i, name in enumerate(feature_names)}
    n = X.shape[0]

    def col(name: str) -> np.ndarray:
        return X[:, idx[name]] if name in idx else np.zeros(n)

    trend_strength = col("trend_strength")
    trend_direction = col("trend_direction")
    return {
        "trending": (trend_strength >= 0.5) & (trend_direction != 0),
        "ranging": (trend_strength < 0.5) | (trend_direction == 0),
        "high_volatility": col("vol_expansion") >= 1.0,
        "low_volatility": col("vol_compression") >= 1.0,
        "london_session": col("session_is_london") >= 1.0,
        "newyork_session": col("session_is_newyork") >= 1.0,
        "asian_session": col("session_is_asian") >= 1.0,
        "sydney_session": col("session_is_sydney") >= 1.0,
    }


def residual_regime_breakdown(
    residuals: np.ndarray, X: np.ndarray, feature_names: Sequence[str], min_samples: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Where does the model perform poorly? MAE/RMSE of residuals restricted
    to each market regime -- lets a reviewer see if error concentrates in a
    specific condition (e.g. high volatility) rather than being uniform."""
    residuals = np.asarray(residuals, dtype=float)
    masks = _regime_masks(X, feature_names)
    out: Dict[str, Dict[str, Any]] = {}
    for regime, mask in masks.items():
        n_samples = int(mask.sum())
        if n_samples < min_samples:
            out[regime] = {"n_samples": n_samples, "mae": None, "rmse": None, "note": "insufficient samples"}
            continue
        r = residuals[mask]
        out[regime] = {"n_samples": n_samples, "mae": float(np.mean(np.abs(r))), "rmse": float(np.sqrt(np.mean(r ** 2)))}
    return out


@dataclass(slots=True)
class ResidualDiagnostics:
    """Complete Phase 3 audit for one target's test-set predictions."""

    target: str
    residual_statistics: Dict[str, float]
    error_distribution: Dict[str, Any]
    qq: QQAnalysis
    normality: NormalityResult
    heteroscedasticity: HeteroscedasticityResult
    durbin_watson: float
    autocorrelation: Dict[int, float]
    regime_breakdown: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target, "residual_statistics": self.residual_statistics,
            "error_distribution": self.error_distribution, "qq": self.qq.to_dict(),
            "normality": self.normality.to_dict(), "heteroscedasticity": self.heteroscedasticity.to_dict(),
            "durbin_watson": None if not np.isfinite(self.durbin_watson) else round(self.durbin_watson, 4),
            "autocorrelation": {k: round(v, 4) for k, v in self.autocorrelation.items()},
            "regime_breakdown": self.regime_breakdown,
        }


def analyze_residuals(
    target: str, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray, feature_names: Sequence[str],
) -> ResidualDiagnostics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    return ResidualDiagnostics(
        target=target, residual_statistics=residual_statistics(y_true, y_pred),
        error_distribution=prediction_error_distribution(y_true, y_pred),
        qq=qq_analysis(residuals), normality=test_normality(residuals),
        heteroscedasticity=breusch_pagan_test(residuals, y_pred), durbin_watson=durbin_watson(residuals),
        autocorrelation=autocorrelation(residuals), regime_breakdown=residual_regime_breakdown(residuals, X, feature_names),
    )
