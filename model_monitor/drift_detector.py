"""Market regime detection, target/residual drift, and the top-level
``DriftDetector`` orchestrator that combines them with ``feature_drift.py``.

Regime classification reads only already-computed ``MarketState.to_dict()``
fields (``trend_*``, ``vol_*``, ``session_*``, ``spread_*``) -- it never
recomputes an indicator or re-detects structure itself, exactly like every
other consumer of ``MarketState`` on this platform.

Target/residual drift compare a single numeric series (actual outcomes, or
prediction errors) between the training baseline and a recent live window.
This is deliberately task-agnostic: the caller supplies whatever numeric
array is meaningful for its task (regression: raw actual values / raw
residuals; classification: encoded class labels / one-minus-true-class-
probability as a "classification residual") -- ``drift_detector.py`` itself
never branches on task type.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import ks_2samp

from .config import MonitorConfig
from .exceptions import InsufficientDataError
from .feature_drift import FeatureDriftDetector, FeatureDriftReport, _severity_label

_EPS = 1e-12

TREND_STATES = ("trending_up", "trending_down", "ranging")
VOLATILITY_STATES = ("high_volatility", "low_volatility", "normal")
SESSIONS = ("sydney", "asian", "london", "newyork", "none")
LIQUIDITY_STATES = ("widened_spread", "normal")


# --------------------------------------------------------------------------- #
# Market regime classification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    """One point-in-time characterization of the market regime, derived
    purely from a ``MarketState.to_dict()`` snapshot."""

    trend_state: str
    volatility_state: str
    session: str
    liquidity_state: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "trend_state": self.trend_state, "volatility_state": self.volatility_state,
            "session": self.session, "liquidity_state": self.liquidity_state,
        }


def classify_regime(state: Dict[str, float], trend_strength_threshold: float = 0.5) -> RegimeSnapshot:
    """Classify one ``MarketState.to_dict()`` snapshot into a
    :class:`RegimeSnapshot`. Missing keys default to a neutral reading
    (never raises -- a partially-populated dict just yields a less
    specific regime)."""
    direction = state.get("trend_direction", 0.0)
    strength = state.get("trend_strength", 0.0)
    if direction > 0 and strength >= trend_strength_threshold:
        trend_state = "trending_up"
    elif direction < 0 and strength >= trend_strength_threshold:
        trend_state = "trending_down"
    else:
        trend_state = "ranging"

    if state.get("vol_expansion", 0.0) >= 1.0:
        volatility_state = "high_volatility"
    elif state.get("vol_compression", 0.0) >= 1.0:
        volatility_state = "low_volatility"
    else:
        volatility_state = "normal"

    session = "none"
    for name in ("sydney", "asian", "london", "newyork"):
        if state.get(f"session_is_{name}", 0.0) >= 1.0:
            session = name
            break

    liquidity_state = (
        "widened_spread"
        if state.get("spread_spike", 0.0) >= 1.0 or state.get("spread_percentile", 0.0) >= 0.9
        else "normal"
    )

    return RegimeSnapshot(
        trend_state=trend_state, volatility_state=volatility_state,
        session=session, liquidity_state=liquidity_state,
    )


def _categorical_distribution(snapshots: Sequence[RegimeSnapshot], dimension: str) -> Dict[str, float]:
    values = [getattr(s, dimension) for s in snapshots]
    counts = Counter(values)
    total = max(len(values), 1)
    return {k: v / total for k, v in counts.items()}


def _total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


@dataclass(slots=True)
class RegimeDriftReport:
    """Comparison of the current market regime distribution against the
    regime distribution represented in the training data."""

    baseline_distribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    current_distribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    dimension_shift: Dict[str, float] = field(default_factory=dict)
    differing_dimensions: List[str] = field(default_factory=list)
    dominant_current_regime: Dict[str, str] = field(default_factory=dict)
    overall_shift: float = 0.0
    shift_detected: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "baseline_distribution": self.baseline_distribution,
            "current_distribution": self.current_distribution,
            "dimension_shift": {k: round(v, 4) for k, v in self.dimension_shift.items()},
            "differing_dimensions": self.differing_dimensions,
            "dominant_current_regime": self.dominant_current_regime,
            "overall_shift": round(self.overall_shift, 4),
            "shift_detected": self.shift_detected,
        }


def detect_regime_drift(
    baseline: Sequence[RegimeSnapshot], current: Sequence[RegimeSnapshot], dimension_threshold: float = 0.3,
) -> RegimeDriftReport:
    if not baseline or not current:
        raise InsufficientDataError("detect_regime_drift() needs at least one baseline and one current snapshot.")
    dimensions = ("trend_state", "volatility_state", "session", "liquidity_state")
    baseline_dist = {d: _categorical_distribution(baseline, d) for d in dimensions}
    current_dist = {d: _categorical_distribution(current, d) for d in dimensions}
    shift = {d: _total_variation_distance(baseline_dist[d], current_dist[d]) for d in dimensions}
    differing = [d for d, v in shift.items() if v > dimension_threshold]
    dominant = {d: max(current_dist[d], key=current_dist[d].get) for d in dimensions}
    overall = float(np.mean(list(shift.values())))
    return RegimeDriftReport(
        baseline_distribution=baseline_dist, current_distribution=current_dist,
        dimension_shift=shift, differing_dimensions=differing, dominant_current_regime=dominant,
        overall_shift=overall, shift_detected=bool(differing),
    )


# --------------------------------------------------------------------------- #
# Target / residual drift -- generic 1-D numeric distribution comparison
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class DistributionShift:
    """Generic drift assessment for a single numeric series (used for both
    target drift and residual drift -- the caller decides which series to
    hand in)."""

    baseline_mean: float
    baseline_std: float
    current_mean: float
    current_std: float
    mean_shift: float
    variance_shift: float
    ks_statistic: float
    ks_pvalue: float
    severity: float
    severity_label: str

    def to_dict(self) -> Dict[str, float]:
        return {
            "baseline_mean": self.baseline_mean, "baseline_std": self.baseline_std,
            "current_mean": self.current_mean, "current_std": self.current_std,
            "mean_shift": round(self.mean_shift, 4), "variance_shift": round(self.variance_shift, 4),
            "ks_statistic": round(self.ks_statistic, 4), "ks_pvalue": round(self.ks_pvalue, 4),
            "severity": round(self.severity, 4), "severity_label": self.severity_label,
        }


def detect_distribution_shift(baseline_values: np.ndarray, current_values: np.ndarray) -> DistributionShift:
    baseline_values = np.asarray(baseline_values, dtype=float)
    current_values = np.asarray(current_values, dtype=float)
    if baseline_values.size < 2 or current_values.size < 2:
        raise InsufficientDataError("detect_distribution_shift() needs >= 2 values in both series.")

    baseline_mean, baseline_std = float(baseline_values.mean()), float(baseline_values.std())
    baseline_std_safe = baseline_std if baseline_std > _EPS else 1.0
    current_mean, current_std = float(current_values.mean()), float(current_values.std())

    mean_shift = abs(current_mean - baseline_mean) / baseline_std_safe
    variance_shift = abs(current_std - baseline_std) / baseline_std_safe
    if np.unique(current_values).size >= 2:
        ks_stat, ks_p = ks_2samp(baseline_values, current_values)
    else:
        ks_stat, ks_p = 0.0, 1.0

    severity = float(np.clip(0.45 * min(mean_shift / 3.0, 1.0) + 0.25 * min(variance_shift / 3.0, 1.0)
                              + 0.30 * float(ks_stat), 0.0, 1.0))
    return DistributionShift(
        baseline_mean=baseline_mean, baseline_std=baseline_std, current_mean=current_mean,
        current_std=current_std, mean_shift=mean_shift, variance_shift=variance_shift,
        ks_statistic=float(ks_stat), ks_pvalue=float(ks_p), severity=severity,
        severity_label=_severity_label(severity),
    )


# --------------------------------------------------------------------------- #
# Top-level orchestrator
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class DriftReport:
    """Everything the health-score/retraining-policy layer needs about
    drift, in one object."""

    feature_drift: FeatureDriftReport
    regime_drift: Optional[RegimeDriftReport]
    target_drift: Optional[DistributionShift]
    residual_drift: Optional[DistributionShift]
    overall_severity: float
    severity_label: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_drift": self.feature_drift.to_dict(),
            "regime_drift": self.regime_drift.to_dict() if self.regime_drift else None,
            "target_drift": self.target_drift.to_dict() if self.target_drift else None,
            "residual_drift": self.residual_drift.to_dict() if self.residual_drift else None,
            "overall_severity": round(self.overall_severity, 4),
            "severity_label": self.severity_label,
        }


class DriftDetector:
    """Fits a full drift baseline (features + regime + target + residual)
    once, then scores any number of live batches against it."""

    def __init__(self, config: Optional[MonitorConfig] = None) -> None:
        self.config = config or MonitorConfig()
        self.feature_drift_detector = FeatureDriftDetector(outlier_z_threshold=self.config.outlier_z_threshold)
        self._baseline_regimes: List[RegimeSnapshot] = []
        self._baseline_targets: Optional[np.ndarray] = None
        self._baseline_residuals: Optional[np.ndarray] = None
        self._fitted = False

    def fit_baseline(
        self, X_train: np.ndarray, feature_names: Sequence[str], *,
        regime_snapshots: Optional[Sequence[RegimeSnapshot]] = None,
        target_values: Optional[np.ndarray] = None,
        residual_values: Optional[np.ndarray] = None,
    ) -> "DriftDetector":
        self.feature_drift_detector.fit_baseline(X_train, feature_names)
        self._baseline_regimes = list(regime_snapshots) if regime_snapshots else []
        self._baseline_targets = np.asarray(target_values, dtype=float) if target_values is not None else None
        self._baseline_residuals = np.asarray(residual_values, dtype=float) if residual_values is not None else None
        self._fitted = True
        return self

    def detect(
        self, X_live: np.ndarray, feature_names: Sequence[str], *,
        valid_mask: Optional[np.ndarray] = None,
        current_regimes: Optional[Sequence[RegimeSnapshot]] = None,
        current_targets: Optional[np.ndarray] = None,
        current_residuals: Optional[np.ndarray] = None,
    ) -> DriftReport:
        if not self._fitted:
            raise InsufficientDataError("Call fit_baseline() before detect().")

        feature_report = self.feature_drift_detector.detect(X_live, feature_names, valid_mask=valid_mask)

        regime_report = None
        if current_regimes and self._baseline_regimes:
            regime_report = detect_regime_drift(self._baseline_regimes, current_regimes)

        target_shift = None
        if current_targets is not None and self._baseline_targets is not None:
            target_shift = detect_distribution_shift(self._baseline_targets, current_targets)

        residual_shift = None
        if current_residuals is not None and self._baseline_residuals is not None:
            residual_shift = detect_distribution_shift(self._baseline_residuals, current_residuals)

        components = [feature_report.overall_severity]
        if regime_report is not None:
            components.append(regime_report.overall_shift)
        if target_shift is not None:
            components.append(target_shift.severity)
        if residual_shift is not None:
            components.append(residual_shift.severity)
        overall = float(np.mean(components))

        return DriftReport(
            feature_drift=feature_report, regime_drift=regime_report, target_drift=target_shift,
            residual_drift=residual_shift, overall_severity=overall, severity_label=_severity_label(overall),
        )
