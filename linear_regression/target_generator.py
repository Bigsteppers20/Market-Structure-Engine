"""Regression target computation.

Reuses ``ml_pipeline.label_generator.REGRESSION_REGISTRY`` directly for every
target it already defines (``next_close``, ``next_high``, ``next_low``,
``next_return``, ``expected_pip_movement``, ``future_volatility``, ...) --
zero duplication of that logic. Only the targets genuinely new to this
engine (MFE, MAE, average future price, future range, future midpoint) are
implemented here.

Like ``ml_pipeline.label_generator``, every function here reads
``df.iloc[index+1 : index+1+horizon]`` quite deliberately -- these are
*training labels*, which must look at the future by definition. This has no
bearing on the model's *input*, which is exclusively ``MarketState`` (see
``feature_mapper.py``) -- the two are computed from disjoint slices of the
underlying data, exactly as in ``ml_pipeline``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from ml_pipeline.label_generator import REGRESSION_REGISTRY as _ML_PIPELINE_TARGETS

from .exceptions import UnsupportedTargetError

if TYPE_CHECKING:
    from ml_pipeline.dataset_builder import Dataset

RegressionTargetFn = Callable[[pd.DataFrame, int, int, float], float]


def _future_window(df: pd.DataFrame, index: int, horizon: int) -> pd.DataFrame:
    end = min(index + 1 + horizon, len(df))
    return df.iloc[index + 1: end]


def _maximum_favorable_excursion(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    """Best-case favorable move (long-bias convention): highest high reached
    within the horizon, minus the entry close."""
    window = _future_window(df, index, horizon)
    if window.empty:
        return 0.0
    entry = float(df["close"].iloc[index])
    return float(window["high"].max() - entry)


def _maximum_adverse_excursion(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    """Worst-case adverse move (long-bias convention): entry close minus the
    lowest low reached within the horizon (a positive magnitude)."""
    window = _future_window(df, index, horizon)
    if window.empty:
        return 0.0
    entry = float(df["close"].iloc[index])
    return float(entry - window["low"].min())


def _average_future_price(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    window = _future_window(df, index, horizon)
    if window.empty:
        return float(df["close"].iloc[index])
    return float(window["close"].mean())


def _future_range(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    window = _future_window(df, index, horizon)
    if window.empty:
        return 0.0
    return float(window["high"].max() - window["low"].min())


def _future_midpoint(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    window = _future_window(df, index, horizon)
    if window.empty:
        return float(df["close"].iloc[index])
    return float((window["high"].max() + window["low"].min()) / 2.0)


#: New targets not already covered by ml_pipeline's registry.
_NEW_TARGETS: Dict[str, RegressionTargetFn] = {
    "maximum_favorable_excursion": _maximum_favorable_excursion,
    "maximum_adverse_excursion": _maximum_adverse_excursion,
    "average_future_price": _average_future_price,
    "future_range": _future_range,
    "future_midpoint": _future_midpoint,
}

#: Full registry available to this engine: every ml_pipeline regression
#: target (reused, not duplicated) plus the 5 new ones above.
REGRESSION_TARGET_REGISTRY: Dict[str, RegressionTargetFn] = {**_ML_PIPELINE_TARGETS, **_NEW_TARGETS}

#: Maps a target name to the corresponding field on
#: ``predictor.RegressionPrediction`` -- only targets with a named slot in
#: the spec's output object are listed; any other registered target is
#: still trainable/predictable, just not auto-mapped onto a dedicated field.
TARGET_TO_PREDICTION_FIELD: Dict[str, str] = {
    "next_close": "expected_close",
    "next_high": "expected_high",
    "next_low": "expected_low",
    "next_return": "expected_return",
    "expected_pip_movement": "expected_pip_move",
    "future_volatility": "expected_volatility",
    "maximum_favorable_excursion": "expected_MFE",
    "maximum_adverse_excursion": "expected_MAE",
}


def compute_target(df: pd.DataFrame, index: int, horizon: int, target: str, pip_size: float = 0.0001) -> float:
    if target not in REGRESSION_TARGET_REGISTRY:
        raise UnsupportedTargetError(
            f"Unknown regression target {target!r}. Registered: {sorted(REGRESSION_TARGET_REGISTRY)}"
        )
    return REGRESSION_TARGET_REGISTRY[target](df, index, horizon, pip_size)


def compute_targets(
    df: pd.DataFrame, index: int, horizon: int, targets: Sequence[str], pip_size: float = 0.0001
) -> Dict[str, float]:
    return {t: compute_target(df, index, horizon, t, pip_size) for t in targets}


def augment_dataset_targets(
    dataset: "Dataset", raw_df: pd.DataFrame, targets: Sequence[str], horizon: int, pip_size: float = 0.0001
) -> Dict[str, np.ndarray]:
    """Compute targets *not* natively supported by ``ml_pipeline.DatasetBuilder``
    (i.e. the 5 new ones in :data:`_NEW_TARGETS`) for an already-built
    ``Dataset``, without modifying or reimplementing ``DatasetBuilder`` itself.

    ``ml_pipeline.DatasetConfig.regression_targets`` is validated against a
    fixed whitelist that only covers ``ml_pipeline``'s own registry, so
    ``DatasetBuilder.build()`` cannot compute MFE/MAE/average future
    price/future range/future midpoint directly. Instead, this function
    matches each sample's decision-bar timestamp (already present in
    ``dataset.metadata``, computed by the unmodified, leakage-safe
    ``DatasetBuilder``) back to its index in the original raw historical
    DataFrame, and computes the new targets at exactly those same decision
    points -- inheriting the same leakage guarantees (each target function
    only reads bars strictly after its own index) without duplicating any
    of ``DatasetBuilder``'s rolling-window logic.

    Raises
    ------
    ValueError
        If a decision timestamp from ``dataset.metadata`` cannot be found in
        ``raw_df`` -- almost always means ``raw_df`` isn't the exact same
        historical data that was passed to ``DatasetBuilder.build()``.
    """
    raw_df = raw_df.reset_index(drop=True)
    timestamp_to_index = {ts: i for i, ts in enumerate(pd.to_datetime(raw_df["timestamp"]))}
    decision_timestamps = pd.to_datetime(dataset.metadata["timestamp"])

    out: Dict[str, List[float]] = {t: [] for t in targets}
    for ts in decision_timestamps:
        index = timestamp_to_index.get(ts)
        if index is None:
            raise ValueError(
                f"Decision timestamp {ts} from the built Dataset was not found in raw_df -- "
                "raw_df must be the exact same historical data passed to DatasetBuilder.build()."
            )
        for target in targets:
            out[target].append(compute_target(raw_df, index, horizon, target, pip_size))
    return {t: np.asarray(v, dtype=np.float64) for t, v in out.items()}
