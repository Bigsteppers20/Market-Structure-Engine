"""Feature-level drift detection: current live MarketState feature vectors
against the training feature distribution.

Model-agnostic by construction: both the Linear Regression Engine and the
Logistic Regression Engine (and any future model) consume the identical
185-dimensional ``MarketState.to_vector()`` feature space, so this module
never needs to know which model produced a prediction -- it only ever
operates on raw feature matrices + names.

Detects (per the spec's FEATURE DRIFT section):

* Distribution drift -- population stability index (PSI) + a two-sample
  Kolmogorov-Smirnov test per feature.
* Mean shift / variance shift -- in training-standard-deviation units.
* Feature correlation drift -- mean absolute change in the pairwise
  correlation matrix.
* Missing feature drift -- a feature entirely absent from the live schema,
  or (when a validity mask is supplied) present but flagged invalid for
  every live row.
* Outlier frequency -- fraction of live rows with |z-score| beyond a
  configurable threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import ks_2samp

from .exceptions import InsufficientDataError

_EPS = 1e-12


@dataclass(slots=True)
class FeatureDriftMetric:
    """Per-feature drift diagnostics."""

    feature: str
    train_mean: float
    train_std: float
    live_mean: float
    live_std: float
    mean_shift: float
    """|live_mean - train_mean| / train_std -- in training-std-dev units."""
    variance_shift: float
    """|live_std - train_std| / train_std."""
    psi: float
    """Population Stability Index -- < 0.1 low, < 0.25 moderate, else severe."""
    ks_statistic: float
    ks_pvalue: float
    outlier_frequency: float
    missing_frequency: float
    drift_score: float
    """Combined per-feature severity in [0, 1]."""

    def to_dict(self) -> Dict[str, float]:
        return {
            "train_mean": self.train_mean, "train_std": self.train_std,
            "live_mean": self.live_mean, "live_std": self.live_std,
            "mean_shift": self.mean_shift, "variance_shift": self.variance_shift,
            "psi": self.psi, "ks_statistic": self.ks_statistic, "ks_pvalue": self.ks_pvalue,
            "outlier_frequency": self.outlier_frequency, "missing_frequency": self.missing_frequency,
            "drift_score": self.drift_score,
        }


@dataclass(slots=True)
class FeatureDriftReport:
    """Full feature-drift assessment for one batch of live feature vectors."""

    per_feature: Dict[str, FeatureDriftMetric] = field(default_factory=dict)
    most_drifted: List[str] = field(default_factory=list)
    missing_features: List[str] = field(default_factory=list)
    unexpected_features: List[str] = field(default_factory=list)
    correlation_drift: float = 0.0
    overall_severity: float = 0.0
    severity_label: str = "low"
    n_live_samples: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "per_feature": {name: m.to_dict() for name, m in self.per_feature.items()},
            "most_drifted": self.most_drifted,
            "missing_features": self.missing_features,
            "unexpected_features": self.unexpected_features,
            "correlation_drift": round(self.correlation_drift, 4),
            "overall_severity": round(self.overall_severity, 4),
            "severity_label": self.severity_label,
            "n_live_samples": self.n_live_samples,
        }


def _severity_label(score: float) -> str:
    if score < 0.15:
        return "low"
    if score < 0.35:
        return "moderate"
    if score < 0.6:
        return "high"
    return "severe"


def _psi(train_values: np.ndarray, live_values: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using training-quantile bin edges."""
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(train_values, quantiles))
    if edges.size < 2:
        return 0.0
    train_counts, _ = np.histogram(train_values, bins=edges)
    live_counts, _ = np.histogram(live_values, bins=edges)
    train_pct = train_counts / max(train_counts.sum(), 1)
    live_pct = live_counts / max(live_counts.sum(), 1)
    train_pct = np.clip(train_pct, _EPS, None)
    live_pct = np.clip(live_pct, _EPS, None)
    return float(np.sum((live_pct - train_pct) * np.log(live_pct / train_pct)))


class FeatureDriftDetector:
    """Fits a training-feature baseline once, then scores any number of
    live batches against it."""

    def __init__(self, outlier_z_threshold: float = 3.0, psi_bins: int = 10, top_n: int = 10) -> None:
        self.outlier_z_threshold = outlier_z_threshold
        self.psi_bins = psi_bins
        self.top_n = top_n
        self._feature_names: List[str] = []
        self._train_mean: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None
        self._train_columns: Optional[np.ndarray] = None
        self._train_corr: Optional[np.ndarray] = None
        self._fitted = False

    def fit_baseline(self, X_train: np.ndarray, feature_names: Sequence[str]) -> "FeatureDriftDetector":
        X_train = np.asarray(X_train, dtype=float)
        if X_train.shape[0] < 2:
            raise InsufficientDataError("fit_baseline() needs >= 2 training rows to compute a distribution.")
        self._feature_names = list(feature_names)
        self._train_mean = X_train.mean(axis=0)
        self._train_std = np.where(X_train.std(axis=0) < _EPS, 1.0, X_train.std(axis=0))
        self._train_columns = X_train
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(X_train, rowvar=False)
        self._train_corr = np.nan_to_num(corr, nan=0.0)
        self._fitted = True
        return self

    def detect(
        self, X_live: np.ndarray, feature_names: Sequence[str], valid_mask: Optional[np.ndarray] = None,
    ) -> FeatureDriftReport:
        """``valid_mask`` (optional): boolean array shaped like ``X_live``,
        ``True`` where that feature's ``_valid`` companion flag was 1.0 --
        used for missing-feature-drift (a feature present in the schema but
        invalid for every live row)."""
        if not self._fitted:
            raise InsufficientDataError("Call fit_baseline() before detect().")
        X_live = np.asarray(X_live, dtype=float)
        if X_live.ndim == 1:
            X_live = X_live.reshape(1, -1)

        live_names = list(feature_names)
        missing = [n for n in self._feature_names if n not in live_names]
        unexpected = [n for n in live_names if n not in self._feature_names]
        live_index = {n: i for i, n in enumerate(live_names)}

        per_feature: Dict[str, FeatureDriftMetric] = {}
        assert self._train_mean is not None and self._train_std is not None and self._train_columns is not None
        for j, name in enumerate(self._feature_names):
            if name not in live_index:
                continue
            i = live_index[name]
            train_col = self._train_columns[:, j]
            live_col = X_live[:, i]
            train_mean, train_std = float(self._train_mean[j]), float(self._train_std[j])
            live_mean = float(live_col.mean())
            live_std = float(live_col.std()) if live_col.size > 1 else 0.0

            mean_shift = abs(live_mean - train_mean) / train_std
            variance_shift = abs(live_std - train_std) / train_std
            psi = _psi(train_col, live_col, bins=self.psi_bins)
            if live_col.size >= 2 and np.unique(live_col).size >= 2:
                ks_stat, ks_p = ks_2samp(train_col, live_col)
            else:
                ks_stat, ks_p = 0.0, 1.0

            z_scores = np.abs((live_col - train_mean) / train_std)
            outlier_frequency = float(np.mean(z_scores > self.outlier_z_threshold))
            missing_frequency = 0.0
            if valid_mask is not None:
                valid_mask = np.asarray(valid_mask)
                col_mask = valid_mask[:, i] if valid_mask.ndim == 2 else valid_mask
                missing_frequency = float(np.mean(~col_mask.astype(bool)))

            drift_score = float(np.clip(
                0.30 * min(mean_shift / 3.0, 1.0)
                + 0.20 * min(variance_shift / 3.0, 1.0)
                + 0.25 * min(psi / 0.25, 1.0)
                + 0.10 * outlier_frequency
                + 0.15 * missing_frequency,
                0.0, 1.0,
            ))

            per_feature[name] = FeatureDriftMetric(
                feature=name, train_mean=train_mean, train_std=train_std,
                live_mean=live_mean, live_std=live_std, mean_shift=mean_shift,
                variance_shift=variance_shift, psi=psi, ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_p), outlier_frequency=outlier_frequency,
                missing_frequency=missing_frequency, drift_score=drift_score,
            )

        for name in missing:
            per_feature[name] = FeatureDriftMetric(
                feature=name, train_mean=float(self._train_mean[self._feature_names.index(name)]),
                train_std=float(self._train_std[self._feature_names.index(name)]),
                live_mean=0.0, live_std=0.0, mean_shift=0.0, variance_shift=0.0, psi=0.0,
                ks_statistic=0.0, ks_pvalue=1.0, outlier_frequency=0.0, missing_frequency=1.0,
                drift_score=1.0,
            )

        most_drifted = [
            n for n, _ in sorted(per_feature.items(), key=lambda kv: kv[1].drift_score, reverse=True)
        ][: self.top_n]

        correlation_drift = 0.0
        common = [n for n in self._feature_names if n in live_index]
        if len(common) >= 2 and self._train_corr is not None:
            train_idx = [self._feature_names.index(n) for n in common]
            live_idx = [live_index[n] for n in common]
            with np.errstate(invalid="ignore"):
                live_corr = np.nan_to_num(np.corrcoef(X_live[:, live_idx], rowvar=False), nan=0.0)
            train_sub = self._train_corr[np.ix_(train_idx, train_idx)]
            iu = np.triu_indices(len(common), k=1)
            if iu[0].size:
                correlation_drift = float(np.mean(np.abs(live_corr[iu] - train_sub[iu])) / 2.0)

        mean_feature_drift = float(np.mean([m.drift_score for m in per_feature.values()])) if per_feature else 0.0
        overall_severity = float(np.clip(0.8 * mean_feature_drift + 0.2 * min(correlation_drift * 2.0, 1.0), 0.0, 1.0))

        return FeatureDriftReport(
            per_feature=per_feature, most_drifted=most_drifted, missing_features=missing,
            unexpected_features=unexpected, correlation_drift=correlation_drift,
            overall_severity=overall_severity, severity_label=_severity_label(overall_severity),
            n_live_samples=X_live.shape[0],
        )
