"""Thin wrapper around a scikit-learn linear estimator.

Supports single- or multi-output regression uniformly (scikit-learn's
``LinearRegression``/``Ridge``/``Lasso``/``ElasticNet`` all natively accept a
2-D ``y``), plus an optional bootstrap ensemble used purely to estimate
per-prediction uncertainty (fed into ``confidence.py`` and the prediction
interval) -- never used to change the point estimate itself.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from .exceptions import ModelNotTrainedError, UnsupportedModelTypeError

MODEL_TYPES: Dict[str, type] = {
    "linear": LinearRegression, "ridge": Ridge, "lasso": Lasso, "elasticnet": ElasticNet,
}


class RegressionModel:
    """A fitted (or fittable) linear regression model plus a bootstrap
    ensemble for uncertainty estimation.

    Parameters
    ----------
    model_type:
        One of ``"linear"``, ``"ridge"``, ``"lasso"``, ``"elasticnet"``.
    n_bootstrap:
        Number of bootstrap-resampled estimators to fit alongside the
        primary one (0 disables uncertainty estimation).
    random_state:
        Seed for the bootstrap resampling.
    """

    def __init__(self, model_type: str = "linear", n_bootstrap: int = 10,
                 random_state: int = 42, **hyperparameters: Any) -> None:
        if model_type not in MODEL_TYPES:
            raise UnsupportedModelTypeError(f"model_type={model_type!r}, expected one of {sorted(MODEL_TYPES)}.")
        self.model_type = model_type
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.hyperparameters = hyperparameters
        self._estimator = MODEL_TYPES[model_type](**hyperparameters)
        self._bootstrap_estimators: List[Any] = []
        self.target_names_: List[str] = []
        self.n_outputs_: int = 0
        self._fitted = False

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray, target_names: List[str]) -> "RegressionModel":
        self.target_names_ = list(target_names)
        y = np.asarray(y, dtype=float)
        self.n_outputs_ = y.shape[1] if y.ndim == 2 else 1
        self._estimator.fit(X, y)

        self._bootstrap_estimators = []
        if self.n_bootstrap > 0:
            rng = np.random.default_rng(self.random_state)
            n = X.shape[0]
            for _ in range(self.n_bootstrap):
                idx = rng.integers(0, n, size=n)
                est = MODEL_TYPES[self.model_type](**self.hyperparameters)
                est.fit(X[idx], y[idx])
                self._bootstrap_estimators.append(est)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions shaped ``(n_samples, n_outputs)``."""
        if not self._fitted:
            raise ModelNotTrainedError("RegressionModel.fit() must be called before predict().")
        preds = self._estimator.predict(X)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(point_estimate, std)``; ``std`` is ``None`` if no
        bootstrap ensemble was fit. Both shaped ``(n_samples, n_outputs)``."""
        point = self.predict(X)
        if not self._bootstrap_estimators:
            return point, None
        ensemble_preds = np.stack([
            (est.predict(X).reshape(-1, 1) if self.n_outputs_ == 1 else est.predict(X))
            for est in self._bootstrap_estimators
        ])
        return point, ensemble_preds.std(axis=0)

    # ------------------------------------------------------------------ #
    @property
    def coefficients(self) -> Optional[np.ndarray]:
        return getattr(self._estimator, "coef_", None) if self._fitted else None

    @property
    def intercept(self) -> Optional[np.ndarray]:
        return getattr(self._estimator, "intercept_", None) if self._fitted else None

    def feature_importance(self, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Mean absolute coefficient magnitude per feature, averaged across
        outputs for multi-output models -- linear regression's natural,
        directly interpretable notion of feature importance."""
        coef = self.coefficients
        if coef is None:
            return None
        coef = np.atleast_2d(coef)
        importance = np.abs(coef).mean(axis=0)
        return dict(zip(feature_names, (float(v) for v in importance)))
