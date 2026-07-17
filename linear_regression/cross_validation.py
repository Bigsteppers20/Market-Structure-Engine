"""Phase 4 -- time-series cross-validation, and Phase 5 -- model-type
comparison (linear/ridge/lasso/elasticnet), both feeding
``LINEAR_REGRESSION_IMPROVEMENT_REPORT.md``.

Never shuffles data. Walk-forward/rolling/expanding folds are produced
entirely by ``ml_pipeline.splitter.TimeSeriesSplitter`` (reused unmodified,
its own already-leakage-safe walk-forward implementation) -- this module
only orchestrates fitting/scoring across whatever folds that splitter
returns, and never invents its own splitting logic.

Hyperparameter selection (for ridge/lasso/elasticnet) uses an internal,
time-ordered holdout carved out of the TRAINING fold only -- the same
``INTERNAL_HOLDOUT_FRACTION`` convention ``trainer.py`` already uses for its
confidence-supporting statistics -- so no test fold is ever touched during
model selection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from ml_pipeline.splitter import TimeSeriesSplitter
from .metrics import compute_all_regression_metrics
from .regression_model import RegressionModel

#: Sensible per-model-type search grids -- ridge/lasso/elasticnet operate on
#: very different alpha scales (see LINEAR_REGRESSION_IMPROVEMENT_REPORT.md's
#: regularization findings), so a single shared grid would unfairly
#: penalize whichever family's natural scale doesn't match it.
DEFAULT_ALPHA_GRIDS: Dict[str, List[float]] = {
    "linear": [],
    "ridge": [1.0, 10.0, 50.0, 100.0, 300.0, 1000.0],
    "lasso": [0.0001, 0.001, 0.01, 0.1, 1.0],
    "elasticnet": [0.0001, 0.001, 0.01, 0.1, 1.0],
}

INTERNAL_HOLDOUT_FRACTION = 0.15


#: Coordinate-descent solvers (lasso/elasticnet) need more than sklearn's
#: default 1000 iterations to fully converge on 185 correlated features at
#: the small alpha values these targets sometimes favor -- a call-site
#: hyperparameter choice, not a change to RegressionModel/RegressionConfig.
_EXTRA_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "lasso": {"max_iter": 5000}, "elasticnet": {"max_iter": 5000},
}


def _select_hyperparameters(
    X_train: np.ndarray, y_train: np.ndarray, model_type: str, alpha_grid: Sequence[float],
    random_state: int = 42,
) -> Dict[str, float]:
    """Pick the best alpha via an internal, time-ordered holdout carved from
    the training fold -- never the test fold. Returns ``{}`` for
    ``model_type == "linear"`` (no regularization strength to tune)."""
    extra = _EXTRA_HYPERPARAMS.get(model_type, {})
    if not alpha_grid:
        return dict(extra)
    n = X_train.shape[0]
    val_size = max(5, int(n * INTERNAL_HOLDOUT_FRACTION))
    if n - val_size < 10:
        return {"alpha": alpha_grid[len(alpha_grid) // 2], **extra}  # not enough data to search -- middle-of-grid default

    X_fit, X_val = X_train[:-val_size], X_train[-val_size:]
    y_fit, y_val = y_train[:-val_size], y_train[-val_size:]
    best_alpha, best_r2 = alpha_grid[0], -np.inf
    for alpha in alpha_grid:
        model = RegressionModel(model_type=model_type, n_bootstrap=0, random_state=random_state, alpha=alpha, **extra)
        model.fit(X_fit, y_fit, ["_probe"])
        pred = model.predict(X_val).ravel()
        r2 = r2_score(y_val, pred)
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha
    return {"alpha": best_alpha, **extra}


# --------------------------------------------------------------------------- #
# Phase 4 -- cross-validation
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class FoldResult:
    fold_index: int
    n_train: int
    n_test: int
    r2: float
    mae: float
    rmse: float
    explained_variance: float
    hyperparameters: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_index": self.fold_index, "n_train": self.n_train, "n_test": self.n_test,
            "r2": round(self.r2, 4), "mae": round(self.mae, 6), "rmse": round(self.rmse, 6),
            "explained_variance": round(self.explained_variance, 4), "hyperparameters": self.hyperparameters,
        }


@dataclass(slots=True)
class CrossValidationResult:
    target: str
    model_type: str
    method: str
    folds: List[FoldResult]
    mean_r2: float
    std_r2: float
    mean_mae: float
    std_mae: float
    mean_rmse: float
    std_rmse: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target, "model_type": self.model_type, "method": self.method,
            "n_folds": len(self.folds), "folds": [f.to_dict() for f in self.folds],
            "mean_r2": round(self.mean_r2, 4), "std_r2": round(self.std_r2, 4),
            "mean_mae": round(self.mean_mae, 6), "std_mae": round(self.std_mae, 6),
            "mean_rmse": round(self.mean_rmse, 6), "std_rmse": round(self.std_rmse, 6),
        }


def cross_validate(
    X: np.ndarray, y: np.ndarray, *, target_name: str = "target", model_type: str = "ridge",
    method: str = "walk_forward_expanding", n_folds: int = 5, test_size: Optional[int] = None,
    train_size: Optional[int] = None, tune_alpha: bool = True, alpha_grid: Optional[Sequence[float]] = None,
    random_state: int = 42,
) -> CrossValidationResult:
    """Walk-forward time-series cross-validation.

    ``method`` is one of ``"walk_forward_expanding"`` (train grows to
    include every prior fold) or ``"walk_forward_rolling"`` (train is a
    fixed-size sliding window) -- both delegate directly to
    ``TimeSeriesSplitter(method="walk_forward", window_mode=...)``, never
    shuffling the data.
    """
    window_mode = "rolling" if method == "walk_forward_rolling" else "expanding"
    splitter = TimeSeriesSplitter(method="walk_forward", window_mode=window_mode, n_folds=n_folds,
                                   train_size=train_size, test_size=test_size)
    splits = splitter.split(len(y))
    grid = list(alpha_grid) if alpha_grid is not None else DEFAULT_ALPHA_GRIDS.get(model_type, [])

    fold_results: List[FoldResult] = []
    for i, split in enumerate(splits):
        X_train, X_test = X[split.train_idx], X[split.test_idx]
        y_train, y_test = y[split.train_idx], y[split.test_idx]
        if X_test.shape[0] == 0 or X_train.shape[0] < 10:
            continue

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        hyperparams = _select_hyperparameters(X_train_s, y_train, model_type, grid, random_state) if tune_alpha else {}
        model = RegressionModel(model_type=model_type, n_bootstrap=0, random_state=random_state, **hyperparams)
        model.fit(X_train_s, y_train, [target_name])
        pred = model.predict(X_test_s).ravel()
        m = compute_all_regression_metrics(y_test, pred)

        fold_results.append(FoldResult(
            fold_index=i, n_train=int(X_train.shape[0]), n_test=int(X_test.shape[0]),
            r2=m["r2"], mae=m["mae"], rmse=m["rmse"], explained_variance=m["explained_variance"],
            hyperparameters=hyperparams,
        ))

    if not fold_results:
        raise ValueError(f"No usable folds produced for target={target_name!r} -- increase data or reduce n_folds.")

    r2s = [f.r2 for f in fold_results]
    maes = [f.mae for f in fold_results]
    rmses = [f.rmse for f in fold_results]
    return CrossValidationResult(
        target=target_name, model_type=model_type, method=method, folds=fold_results,
        mean_r2=float(np.mean(r2s)), std_r2=float(np.std(r2s)),
        mean_mae=float(np.mean(maes)), std_mae=float(np.std(maes)),
        mean_rmse=float(np.mean(rmses)), std_rmse=float(np.std(rmses)),
    )


# --------------------------------------------------------------------------- #
# Phase 5 -- model-type comparison
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ModelComparisonResult:
    target: str
    model_type: str
    hyperparameters: Dict[str, float]
    mae: float
    rmse: float
    r2: float
    explained_variance: float
    training_time_ms: float
    inference_time_ms: float
    n_nonzero_coefficients: int
    coefficient_l2_norm: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target, "model_type": self.model_type, "hyperparameters": self.hyperparameters,
            "mae": round(self.mae, 6), "rmse": round(self.rmse, 6), "r2": round(self.r2, 4),
            "explained_variance": round(self.explained_variance, 4),
            "training_time_ms": round(self.training_time_ms, 2), "inference_time_ms": round(self.inference_time_ms, 3),
            "n_nonzero_coefficients": self.n_nonzero_coefficients, "coefficient_l2_norm": round(self.coefficient_l2_norm, 4),
        }


def compare_model_types(
    X: np.ndarray, y: np.ndarray, split, *, target_name: str = "target",
    model_types: Sequence[str] = ("linear", "ridge", "lasso", "elasticnet"),
    alpha_grids: Optional[Dict[str, Sequence[float]]] = None, random_state: int = 42,
) -> List[ModelComparisonResult]:
    """Train + score every requested model type on the *same* train/test
    split (single held-out comparison, complementary to the walk-forward
    ``cross_validate()`` above) -- reports MAE/RMSE/R2/explained variance,
    wall-clock training/inference time, and model complexity (nonzero
    coefficient count + L2 norm, as a proxy for how much the model relies
    on many small, unstable coefficients vs. a few strong ones)."""
    grids = alpha_grids or DEFAULT_ALPHA_GRIDS
    scaler = StandardScaler().fit(X[split.train_idx])
    X_train, X_test = scaler.transform(X[split.train_idx]), scaler.transform(X[split.test_idx])
    y_train, y_test = y[split.train_idx], y[split.test_idx]

    results: List[ModelComparisonResult] = []
    for model_type in model_types:
        grid = grids.get(model_type, [])
        hyperparams = _select_hyperparameters(X_train, y_train, model_type, grid, random_state)

        t0 = time.perf_counter()
        model = RegressionModel(model_type=model_type, n_bootstrap=0, random_state=random_state, **hyperparams)
        model.fit(X_train, y_train, [target_name])
        training_time_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        pred = model.predict(X_test).ravel()
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        m = compute_all_regression_metrics(y_test, pred)
        coef = model.coefficients
        coef = np.ravel(coef) if coef is not None else np.array([])
        n_nonzero = int(np.sum(np.abs(coef) > 1e-6))
        l2_norm = float(np.linalg.norm(coef))

        results.append(ModelComparisonResult(
            target=target_name, model_type=model_type, hyperparameters=hyperparams,
            mae=m["mae"], rmse=m["rmse"], r2=m["r2"], explained_variance=m["explained_variance"],
            training_time_ms=training_time_ms, inference_time_ms=inference_time_ms,
            n_nonzero_coefficients=n_nonzero, coefficient_l2_norm=l2_norm,
        ))
    return results


def recommend_best_model(results: Sequence[ModelComparisonResult]) -> str:
    """Best model by test R2 (ties broken by lower RMSE) -- purely
    descriptive, not used to auto-configure anything."""
    best = max(results, key=lambda r: (r.r2, -r.rmse))
    return best.model_type
