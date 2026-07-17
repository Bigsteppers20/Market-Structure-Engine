"""Feature scaling, fit on training data only.

Wraps scikit-learn's ``StandardScaler``/``MinMaxScaler``/``RobustScaler``
(the exact classes named in the project spec) behind one small interface,
plus a ``"none"`` identity option. The scaler is always fit on the training
split alone and reused (never refit) on validation/test/inference data --
:meth:`FeatureScaler.fit` is a hard boundary, not just a convention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class FeatureScaler:
    """Fit-once, reuse-everywhere feature scaler.

    Parameters
    ----------
    method:
        One of ``"standard"``, ``"minmax"``, ``"robust"``, ``"none"``.
    """

    def __init__(self, method: str = "standard") -> None:
        if method not in (*_SCALERS, "none"):
            raise ValueError(f"Unknown scaler method {method!r}. Choose from {[*_SCALERS, 'none']}.")
        self.method = method
        self._scaler = _SCALERS[method]() if method != "none" else None
        self._fitted = False
        self.n_features_: Optional[int] = None

    def fit(self, X_train: np.ndarray) -> "FeatureScaler":
        """Fit on the training split only. Never call this again on val/test data."""
        self.n_features_ = X_train.shape[1]
        if self._scaler is not None:
            self._scaler.fit(X_train)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureScaler.fit() must be called before transform().")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]} -- "
                "was this scaler fit on a different feature set?"
            )
        if self._scaler is None:
            return X.astype(np.float64, copy=True)
        return self._scaler.transform(X)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            return X.astype(np.float64, copy=True)
        return self._scaler.inverse_transform(X)

    def save(self, path: str | Path) -> None:
        """Persist the fitted scaler for later inference (joblib format)."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted FeatureScaler.")
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}.")
        return obj
