"""Phase 8 -- Decision Engine metadata: model-level 0-100 health scores.

These are training-time constants (one set per fitted model, attached to
the ``RegressionModel`` and threaded through every prediction it makes) --
distinct from ``confidence.py``'s per-prediction factors
(``prediction_stability``/``distribution_distance``, which genuinely vary
per input and stay computed in ``predictor.py`` at inference time).

Every score here is derived only from quantities the trainer already
computes cheaply (in-sample vs. holdout R², residual scale, feature-target
correlation) -- computing a full walk-forward cross-validation or feature
audit on every training run would change this engine's existing training
cost/behavior, which the "do not modify existing APIs/behavior" constraint
forbids doing unconditionally. Cross-validation awareness is strictly
opt-in (``RegressionConfig.enable_cross_validation``, defaulting to
``False`` -- see ``trainer.py``) and simply reports a neutral default when
not supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

#: Reported when the underlying quantity wasn't computed for this model
#: (e.g. cross-validation wasn't enabled) -- neutral, not a penalty or a
#: reward, same convention as ``logistic_regression.confidence``'s
#: ``calibration_quality`` default.
_NEUTRAL_DEFAULT = 60.0


def _clip(value: float) -> float:
    return float(max(0.0, min(100.0, value)))


def generalization_score(train_r2: float, holdout_r2: float) -> float:
    """100 when the holdout (out-of-sample) R2 matches or exceeds the
    in-sample R2 (no overfitting gap); degrades toward 0 as the gap grows.
    A large positive gap (train >> holdout) is the textbook overfitting
    signature this score is built to catch."""
    if train_r2 <= 0 and holdout_r2 <= 0:
        return _NEUTRAL_DEFAULT
    gap = train_r2 - holdout_r2
    return _clip(100.0 * (1.0 - min(max(gap, 0.0) / 0.5, 1.0)))


def cross_validation_score(cv_mean_r2: Optional[float], cv_std_r2: Optional[float]) -> float:
    """Blends CV mean R2 (higher = better) and CV std R2 (lower = more
    stable across folds -- a high mean with high variance is a lucky
    single fold, not genuine generalization). ``None`` (CV not run) reports
    the neutral default, never a penalty for not opting in."""
    if cv_mean_r2 is None or cv_std_r2 is None:
        return _NEUTRAL_DEFAULT
    mean_component = _clip(max(0.0, min(cv_mean_r2, 1.0)) * 100.0)
    stability_component = _clip(100.0 * (1.0 - min(cv_std_r2 / 0.5, 1.0)))
    return _clip(0.6 * mean_component + 0.4 * stability_component)


def feature_quality_score(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """Cheap proxy (no full VIF/near-constant sweep -- see module docstring):
    blends the fraction of training features with real variance (not
    near-constant) and the mean |Pearson correlation| of each feature
    against this specific target (a quick signal-density read)."""
    variances = X_train.var(axis=0, ddof=0)
    non_constant_fraction = float(np.mean(variances > 1e-8))

    y_std = float(y_train.std())
    if y_std < 1e-12:
        signal_component = 0.0
    else:
        x_std = X_train.std(axis=0, ddof=0)
        valid = x_std > 1e-12
        if not valid.any():
            signal_component = 0.0
        else:
            correlations = np.array([
                np.corrcoef(X_train[:, j], y_train)[0, 1] if valid[j] else 0.0
                for j in range(X_train.shape[1])
            ])
            correlations = np.nan_to_num(correlations, nan=0.0)
            # Mean |r| across ~185 correlated features is naturally small
            # even for a genuinely useful feature set (see
            # TARGET_ANALYSIS_REPORT.md) -- 0.15 is calibrated against this
            # engine's own live-data findings, not an arbitrary guess.
            signal_component = _clip(min(np.mean(np.abs(correlations)) / 0.15, 1.0) * 100.0)

    return _clip(0.5 * (non_constant_fraction * 100.0) + 0.5 * signal_component)


def target_reliability(holdout_r2: float, cv_mean_r2: Optional[float]) -> float:
    """How reliable THIS TARGET's predictions have historically been --
    prefers the cross-validated mean R2 (more robust than a single
    holdout) when available, falling back to the training-time holdout R2."""
    r2 = cv_mean_r2 if cv_mean_r2 is not None else holdout_r2
    return _clip(max(0.0, min(r2, 1.0)) * 100.0)


def model_health_score(
    generalization: float, cross_validation: float, feature_quality: float, target_reliability_score: float,
) -> float:
    """Single composite -- equal blend of the four training-time scores
    above. Purely descriptive (never used to gate training/serving)."""
    return _clip(np.mean([generalization, cross_validation, feature_quality, target_reliability_score]))


@dataclass(slots=True)
class ModelHealthScores:
    """Every Phase 8 training-time score for one fitted model."""

    generalization_score: float
    cross_validation_score: float
    feature_quality_score: float
    target_reliability: float
    model_health_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "generalization_score": round(self.generalization_score, 2),
            "cross_validation_score": round(self.cross_validation_score, 2),
            "feature_quality_score": round(self.feature_quality_score, 2),
            "target_reliability": round(self.target_reliability, 2),
            "model_health_score": round(self.model_health_score, 2),
        }


def compute_model_health_scores(
    *, train_r2: float, holdout_r2: float, X_train: np.ndarray, y_train: np.ndarray,
    cv_mean_r2: Optional[float] = None, cv_std_r2: Optional[float] = None,
) -> ModelHealthScores:
    gen = generalization_score(train_r2, holdout_r2)
    cv = cross_validation_score(cv_mean_r2, cv_std_r2)
    feat = feature_quality_score(X_train, y_train)
    reliability = target_reliability(holdout_r2, cv_mean_r2)
    health = model_health_score(gen, cv, feat, reliability)
    return ModelHealthScores(
        generalization_score=gen, cross_validation_score=cv, feature_quality_score=feat,
        target_reliability=reliability, model_health_score=health,
    )
