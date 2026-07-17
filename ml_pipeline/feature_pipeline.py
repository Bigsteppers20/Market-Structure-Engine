"""Feature preprocessing: missing-value handling, encoding, ordering, validation.

:class:`FeaturePipeline` is fit once on the training split and reused
unchanged for validation/test/inference, guaranteeing identical feature
ordering and identical imputation values everywhere it's applied -- the
same "fit on train only" discipline used by :mod:`scaler`.

Scaling itself is *not* here (see :mod:`scaler`) -- this module only
prepares a fully finite, consistently-ordered, consistently-encoded matrix
for the scaler/selector stages to consume.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import DatasetConfig

# Signed categorical-code features the engine emits as {-1, 0, 1}.
CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "trend_direction",
    "structure_last_bos_direction",
    "structure_last_choch_direction",
    "liq_last_sweep_direction",
    "fvg_nearest_direction",
    "ob_nearest_direction",
)

# (feature name, cyclical period) for sin/cos encoding.
CYCLICAL_FEATURES: Tuple[Tuple[str, int], ...] = (
    ("session_hour", 24),
    ("session_minute", 60),
    ("session_day_of_week", 7),
    ("session_month", 12),
)


class FeaturePipeline:
    """Fits imputation/encoding rules on training data, applies them consistently."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.feature_names_: List[str] = []
        self._impute_pairs: Dict[str, str] = {}  # base_feature -> valid_flag_feature
        self._impute_values: Dict[str, float] = {}
        self._fitted = False

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, feature_names: List[str]) -> "FeaturePipeline":
        """Learn imputation values from training data. Never call on val/test."""
        self.feature_names_ = list(feature_names)
        self._impute_pairs = {}
        self._impute_values = {}
        if self.config.impute_invalid:
            name_to_idx = {n: i for i, n in enumerate(feature_names)}
            for name in feature_names:
                valid_name = f"{name}_valid"
                if valid_name in name_to_idx and not name.endswith("_valid"):
                    self._impute_pairs[name] = valid_name
                    base_idx, valid_idx = name_to_idx[name], name_to_idx[valid_name]
                    valid_mask = X[:, valid_idx] >= 0.5
                    values = X[valid_mask, base_idx]
                    self._impute_values[name] = (
                        float(np.median(values)) if values.size else float(np.median(X[:, base_idx]))
                    )
        self._fitted = True
        return self

    def transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Apply the fitted imputation/encoding/ordering rules to any split."""
        if not self._fitted:
            raise RuntimeError("FeaturePipeline.fit() must be called before transform().")
        X = self._reorder(X, feature_names)
        names = list(self.feature_names_)

        if self.config.impute_invalid:
            X = self._impute(X, names)
        if self.config.cyclical_time_encoding:
            X, names = self._encode_cyclical(X, names)
        if self.config.one_hot_categorical:
            X, names = self._encode_one_hot(X, names)

        self._validate(X)
        return X, names

    def fit_transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        self.fit(X, feature_names)
        return self.transform(X, feature_names)

    # ------------------------------------------------------------------ #
    def _reorder(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        if feature_names == self.feature_names_:
            return X
        missing = set(self.feature_names_) - set(feature_names)
        if missing:
            raise ValueError(f"Input is missing feature(s) seen during fit: {sorted(missing)}")
        df = pd.DataFrame(X, columns=feature_names)
        return df[self.feature_names_].to_numpy(dtype=np.float64)

    def _impute(self, X: np.ndarray, names: List[str]) -> np.ndarray:
        X = X.copy()
        idx = {n: i for i, n in enumerate(names)}
        for base_name, valid_name in self._impute_pairs.items():
            if base_name not in idx or valid_name not in idx:
                continue
            base_idx, valid_idx = idx[base_name], idx[valid_name]
            invalid_mask = X[:, valid_idx] < 0.5
            if invalid_mask.any():
                X[invalid_mask, base_idx] = self._impute_values[base_name]
        return X

    @staticmethod
    def _encode_cyclical(X: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
        idx = {n: i for i, n in enumerate(names)}
        keep_mask = np.ones(len(names), dtype=bool)
        extra_cols: List[np.ndarray] = []
        extra_names: List[str] = []
        for name, period in CYCLICAL_FEATURES:
            if name not in idx:
                continue
            col = X[:, idx[name]]
            keep_mask[idx[name]] = False
            angle = 2.0 * np.pi * col / period
            extra_cols.append(np.sin(angle))
            extra_names.append(f"{name}_sin")
            extra_cols.append(np.cos(angle))
            extra_names.append(f"{name}_cos")
        new_names = [n for n, keep in zip(names, keep_mask) if keep] + extra_names
        new_X = np.column_stack([X[:, keep_mask]] + extra_cols) if extra_cols else X
        return new_X, new_names

    @staticmethod
    def _encode_one_hot(X: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
        idx = {n: i for i, n in enumerate(names)}
        keep_mask = np.ones(len(names), dtype=bool)
        extra_cols: List[np.ndarray] = []
        extra_names: List[str] = []
        for name in CATEGORICAL_FEATURES:
            if name not in idx:
                continue
            col = X[:, idx[name]]
            keep_mask[idx[name]] = False
            for code, suffix in ((-1.0, "neg"), (0.0, "zero"), (1.0, "pos")):
                extra_cols.append((np.isclose(col, code)).astype(np.float64))
                extra_names.append(f"{name}_{suffix}")
        new_names = [n for n, keep in zip(names, keep_mask) if keep] + extra_names
        new_X = np.column_stack([X[:, keep_mask]] + extra_cols) if extra_cols else X
        return new_X, new_names

    @staticmethod
    def _validate(X: np.ndarray) -> None:
        if not np.isfinite(X).all():
            bad = int((~np.isfinite(X)).sum())
            raise ValueError(
                f"FeaturePipeline output contains {bad} non-finite value(s) -- "
                "this should be impossible after imputation; check for a misconfigured "
                "impute_invalid=False alongside engine features that can be NaN/Inf."
            )
