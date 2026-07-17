"""Thin wrapper around scikit-learn's ``LogisticRegression``.

Handles class balancing (resampling or ``class_weight``), an optional
calibration layer (via ``calibration.py``, using a held-out slice --
never the same rows the base estimator was fit on), and an optional
bootstrap ensemble used purely for prediction-stability confidence and
(indirectly) coefficient-stability diagnostics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .calibration import calibrate_estimator
from .exceptions import ModelNotTrainedError, UnsupportedBalancingStrategyError

BALANCING_STRATEGIES = ("none", "class_weight", "oversample", "undersample", "balanced_sampling")


def resample(X: np.ndarray, y: np.ndarray, strategy: str, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Resample ``(X, y)`` for class balance. Never touches rows outside the
    given arrays (safe to call on a training split only) and never
    reorders across a time boundary the caller didn't already cross --
    logistic regression fitting is order-independent, so shuffling within
    the training split introduces no leakage."""
    if strategy not in BALANCING_STRATEGIES:
        raise UnsupportedBalancingStrategyError(f"Unknown balancing strategy {strategy!r}.")
    if strategy in ("none", "class_weight"):
        return X, y  # class_weight is applied at the estimator level instead

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if strategy == "oversample":
        target = int(counts.max())
    elif strategy == "undersample":
        target = int(counts.min())
    else:  # balanced_sampling
        target = int(round(counts.mean()))

    chosen_idx = []
    for cls in classes:
        idx = np.nonzero(y == cls)[0]
        if idx.size == 0:
            continue
        replace = target > idx.size
        chosen_idx.append(rng.choice(idx, size=target, replace=replace))
    all_idx = np.concatenate(chosen_idx)
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]


class ClassificationModel:
    """A fitted (or fittable) logistic regression classifier, optionally
    calibrated and/or bootstrap-ensembled.

    Parameters
    ----------
    class_balancing:
        One of ``"none"``, ``"class_weight"``, ``"oversample"``,
        ``"undersample"``, ``"balanced_sampling"``.
    calibration_method:
        One of ``"none"``, ``"platt"``, ``"isotonic"``.
    calibration_holdout_fraction:
        Fraction of the (chronologically-ordered) training data reserved as
        an internal calibration holdout -- fit on the remainder, calibrated
        on this slice, never the reverse.
    n_bootstrap:
        Number of bootstrap-resampled classifiers to additionally fit
        (0 disables prediction-stability estimation).
    """

    def __init__(
        self, class_balancing: str = "none", calibration_method: str = "none",
        calibration_holdout_fraction: float = 0.15, n_bootstrap: int = 10,
        random_state: int = 42, **hyperparameters: Any,
    ) -> None:
        self.class_balancing = class_balancing
        self.calibration_method = calibration_method
        self.calibration_holdout_fraction = calibration_holdout_fraction
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.hyperparameters = hyperparameters
        class_weight = "balanced" if class_balancing == "class_weight" else None
        self._base_estimator = LogisticRegression(class_weight=class_weight, max_iter=2000, **hyperparameters)
        self._estimator: Any = self._base_estimator
        self.classes_: List[str] = []
        self.calibration_metadata_: Dict[str, Any] = {"method": "none", "n_calibration_samples": 0}
        self._bootstrap_estimators: List[Any] = []
        self._fitted = False

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray, classes: List[str]) -> "ClassificationModel":
        self.classes_ = list(classes)
        y = np.asarray(y)
        X_bal, y_bal = resample(X, y, self.class_balancing, self.random_state)

        n = X_bal.shape[0]
        val_size = max(5, int(n * self.calibration_holdout_fraction))
        if self.calibration_method != "none" and n - val_size >= 10:
            X_fit, X_cal = X_bal[:-val_size], X_bal[-val_size:]
            y_fit, y_cal = y_bal[:-val_size], y_bal[-val_size:]
        else:
            X_fit, y_fit = X_bal, y_bal
            X_cal = y_cal = None

        self._base_estimator.fit(X_fit, y_fit)
        # Column j of predict_proba() must correspond to encoded label j for
        # `self.classes_[argmax]` (in predict()) to pick the right name --
        # true iff every class appeared in the fitting slice.
        if list(self._base_estimator.classes_) != list(range(len(self.classes_))):
            raise ModelNotTrainedError(
                f"Training data (after balancing) is missing at least one class: "
                f"expected encoded labels {list(range(len(self.classes_)))}, "
                f"got {list(self._base_estimator.classes_)}. Provide more training data "
                "or a class_balancing strategy that guarantees all classes are present."
            )
        if self.calibration_method != "none" and X_cal is not None:
            self._estimator, self.calibration_metadata_ = calibrate_estimator(
                self._base_estimator, X_cal, y_cal, self.calibration_method
            )
        else:
            self._estimator = self._base_estimator
            self.calibration_metadata_ = {"method": "none", "n_calibration_samples": 0}

        self._bootstrap_estimators = []
        if self.n_bootstrap > 0:
            rng = np.random.default_rng(self.random_state)
            for _ in range(self.n_bootstrap):
                idx = rng.integers(0, n, size=n)
                est = LogisticRegression(
                    class_weight=self._base_estimator.class_weight, max_iter=2000, **self.hyperparameters
                )
                boot_X, boot_y = X_bal[idx], y_bal[idx]
                if len(np.unique(boot_y)) < 2:
                    continue  # degenerate bootstrap draw -- skip rather than fail
                est.fit(boot_X, boot_y)
                self._bootstrap_estimators.append(est)

        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ModelNotTrainedError("ClassificationModel.fit() must be called before predict_proba().")
        proba = self._estimator.predict_proba(X)
        # Guard against floating-point drift so rows always sum to exactly 1.
        proba = np.clip(proba, 1e-12, None)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.asarray(self.classes_)[np.argmax(proba, axis=1)]

    def bootstrap_agreement(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Fraction of the bootstrap ensemble agreeing with the primary
        model's predicted (encoded) class, per sample -- ``None`` if no
        ensemble. Both the primary and every bootstrap estimator were fit
        on the same integer-encoded ``y``, so their raw ``predict()``
        outputs are directly comparable without remapping through
        ``self.classes_`` (the string names)."""
        if not self._bootstrap_estimators:
            return None
        primary_encoded = self._estimator.predict(X)
        agreements = np.zeros(X.shape[0])
        for est in self._bootstrap_estimators:
            agreements += (est.predict(X) == primary_encoded).astype(float)
        return agreements / len(self._bootstrap_estimators)

    # ------------------------------------------------------------------ #
    @property
    def coefficients(self) -> Optional[np.ndarray]:
        return getattr(self._base_estimator, "coef_", None) if self._fitted else None

    @property
    def intercept(self) -> Optional[np.ndarray]:
        return getattr(self._base_estimator, "intercept_", None) if self._fitted else None

    @property
    def bootstrap_coefficients(self) -> Optional[List[np.ndarray]]:
        """Per-bootstrap-estimator ``coef_`` arrays, or ``None`` if no
        ensemble was fit -- feeds ``evaluator.build_coefficient_diagnostics``'s
        coefficient-*stability* diagnostic (std of each coefficient across
        the ensemble)."""
        if not self._bootstrap_estimators:
            return None
        return [est.coef_ for est in self._bootstrap_estimators]

    def feature_importance(self, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Mean absolute coefficient magnitude per feature, averaged across
        the one-vs-rest class coefficient rows."""
        coef = self.coefficients
        if coef is None:
            return None
        importance = np.abs(np.atleast_2d(coef)).mean(axis=0)
        return dict(zip(feature_names, (float(v) for v in importance)))
