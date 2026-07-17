"""Optional feature selection.

Five selectable strategies, matching the project spec's names exactly:
``variance`` (VarianceThreshold), ``correlation`` (pairwise Pearson filter),
``mutual_info`` (mutual-information ranking), ``rfe`` (Recursive Feature
Elimination), ``kbest`` (SelectKBest / ANOVA F-test).

Feature selection here is entirely a pre-processing decision about which of
the Market Structure Engine's 185 columns to keep for a specific downstream
model -- it never modifies the engine itself, and nothing selected here is
returned as, or used as, a trained predictive model.

Note on RFE: scikit-learn's `RFE` fundamentally requires a `coef_`-bearing
estimator to rank features by, so this module instantiates a plain
``LinearRegression``/``LogisticRegression`` internally *purely as a feature
ranking tool*. That estimator is never fit on validation/test data, never
returned to the caller, and never used to produce a prediction -- it exists
only inside :meth:`FeatureSelector.fit` to compute a feature ranking, exactly
as RFE requires. This package still builds no predictive model.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

_METHODS = ("variance", "correlation", "mutual_info", "rfe", "kbest")


class FeatureSelector:
    """Fit-once-on-train, reuse-everywhere feature selector.

    Parameters
    ----------
    method:
        One of ``"variance"``, ``"correlation"``, ``"mutual_info"``,
        ``"rfe"``, ``"kbest"``.
    target_type:
        ``"regression"`` or ``"classification"`` -- picks the appropriate
        scoring function for ``mutual_info``/``rfe``/``kbest``. Ignored by
        ``"variance"`` and ``"correlation"`` (unsupervised).
    variance_threshold:
        Minimum variance to keep a feature (for ``"variance"`` only).
    correlation_threshold:
        Absolute-Pearson cutoff above which the later of a pair of features
        is dropped (for ``"correlation"`` only).
    k:
        Number of features to keep (for ``"mutual_info"``, ``"rfe"``,
        ``"kbest"``). Ignored by the unsupervised methods.
    random_state:
        Forwarded to the internal RFE ranking estimator.
    """

    def __init__(
        self,
        method: str,
        target_type: str = "regression",
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.95,
        k: int = 50,
        random_state: int = 42,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"Unknown feature_selector {method!r}. Choose from {_METHODS}.")
        if target_type not in ("regression", "classification"):
            raise ValueError("target_type must be 'regression' or 'classification'.")
        self.method = method
        self.target_type = target_type
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.k = k
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self._mask: Optional[np.ndarray] = None
        self._fit_feature_names: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------ #
    def fit(
        self, X_train: np.ndarray, y_train: Optional[np.ndarray], feature_names: List[str]
    ) -> "FeatureSelector":
        """Fit on the training split only."""
        if self.method == "variance":
            mask = self._fit_variance(X_train)
        elif self.method == "correlation":
            mask = self._fit_correlation(X_train)
        elif self.method == "mutual_info":
            mask = self._fit_mutual_info(X_train, y_train)
        elif self.method == "kbest":
            mask = self._fit_kbest(X_train, y_train)
        else:  # rfe
            mask = self._fit_rfe(X_train, y_train)

        self._mask = mask
        self._fit_feature_names = list(feature_names)
        self.selected_features_ = [n for n, keep in zip(feature_names, mask) if keep]
        self._fitted = True
        return self

    def transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        if not self._fitted:
            raise RuntimeError("FeatureSelector.fit() must be called before transform().")
        if feature_names != self._fit_feature_names:
            # Reorder defensively if given a differently-ordered but equal set.
            idx = [feature_names.index(n) for n in self._fit_feature_names]
            X = X[:, idx]
        return X[:, self._mask], list(self.selected_features_)

    def fit_transform(
        self, X_train: np.ndarray, y_train: Optional[np.ndarray], feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        self.fit(X_train, y_train, feature_names)
        return self.transform(X_train, feature_names)

    # ------------------------------------------------------------------ #
    def _fit_variance(self, X_train: np.ndarray) -> np.ndarray:
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(X_train)
        return vt.get_support()

    def _fit_correlation(self, X_train: np.ndarray) -> np.ndarray:
        """Greedy pairwise-Pearson filter: for any pair above the threshold,
        drop the later column (keep the first-encountered representative)."""
        n_features = X_train.shape[1]
        keep = np.ones(n_features, dtype=bool)
        std = X_train.std(axis=0, ddof=0)
        keep &= std > 1e-12  # drop constant columns outright (also avoids /0 in corrcoef)
        live = np.nonzero(keep)[0]
        corr_live = np.corrcoef(X_train[:, live], rowvar=False) if live.size > 1 else np.zeros((0, 0))
        for a, i in enumerate(live):
            if not keep[i]:
                continue
            for b in range(a + 1, live.size):
                j = live[b]
                if keep[j] and abs(corr_live[a, b]) >= self.correlation_threshold:
                    keep[j] = False
        return keep

    def _fit_mutual_info(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        score_fn = mutual_info_classif if self.target_type == "classification" else mutual_info_regression
        skb = SelectKBest(
            score_func=lambda X, y: score_fn(X, y, random_state=self.random_state),
            k=min(self.k, X_train.shape[1]),
        )
        skb.fit(X_train, y_train)
        return skb.get_support()

    def _fit_kbest(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        score_fn = f_classif if self.target_type == "classification" else f_regression
        skb = SelectKBest(score_func=score_fn, k=min(self.k, X_train.shape[1]))
        skb.fit(X_train, y_train)
        return skb.get_support()

    def _fit_rfe(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        # Ranking-only estimator -- see module docstring.
        estimator = (
            LogisticRegression(max_iter=1000, random_state=self.random_state)
            if self.target_type == "classification"
            else LinearRegression()
        )
        rfe = RFE(estimator=estimator, n_features_to_select=min(self.k, X_train.shape[1]))
        rfe.fit(X_train, y_train)
        return rfe.get_support()
