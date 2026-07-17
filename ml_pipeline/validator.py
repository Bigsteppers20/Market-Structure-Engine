"""Data and dataset validation.

Two layers of checks:

1. **Input validation** (:func:`validate_input_data`) -- run once, before any
   windowing, against the raw historical DataFrame.
2. **Leakage assertion** (:func:`assert_no_future_leakage`) -- run once per
   sample, inside :class:`~ml_pipeline.dataset_builder.DatasetBuilder`'s
   rolling-window loop, turning "the feature window never sees the future"
   from a documentation claim into an enforced runtime invariant.
3. **Dataset validation** (:func:`validate_dataset`) -- run once, after a
   :class:`~ml_pipeline.dataset_builder.Dataset` is assembled, producing the
   :class:`ValidationReport` consumed by ``DATASET_REPORT.md`` generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .dataset_builder import Dataset

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


class LeakageError(RuntimeError):
    """Raised when a feature window would read data at or after the label horizon."""


class DataValidationError(ValueError):
    """Raised when raw input data fails structural validation."""


def validate_input_data(df: pd.DataFrame) -> List[str]:
    """Validate raw historical OHLCV data before any windowing.

    Checks: required columns present, no duplicate timestamps, timestamps
    strictly increasing, no missing candles (gaps in the modal time step).

    Returns a list of human-readable issue strings (empty = clean). Raises
    :class:`DataValidationError` only for structural problems that make
    windowing impossible (missing columns, empty frame); timestamp gaps and
    duplicates are reported as issues, not raised, since the Market
    Structure Engine's own ``DataLoader`` will independently reject
    malformed windows at ``analyze()`` time.
    """
    if df.empty:
        raise DataValidationError("Input DataFrame is empty.")
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {sorted(missing_cols)}")

    issues: List[str] = []
    ts = pd.to_datetime(df["timestamp"])

    dup_count = int(ts.duplicated().sum())
    if dup_count:
        issues.append(f"{dup_count} duplicate timestamp(s) found.")

    if not ts.is_monotonic_increasing:
        issues.append("Timestamps are not strictly increasing (data is unsorted).")

    deltas = ts.diff().dropna()
    if len(deltas) >= 2:
        modal_step = deltas.mode().iloc[0]
        if modal_step > pd.Timedelta(0):
            gap_count = int((deltas > modal_step).sum())
            if gap_count:
                issues.append(
                    f"{gap_count} gap(s) larger than the modal spacing "
                    f"({modal_step}) found -- missing candles."
                )
    return issues


def assert_no_future_leakage(
    decision_index: int, window_start: int, window_end: int, label_index: int
) -> None:
    """Enforce the feature/label leakage boundary for one sample.

    ``window_start..window_end`` is the slice of historical candles handed
    to the Market Structure Engine; ``decision_index`` is the last candle in
    that window (the "as of now" bar); ``label_index`` is where the target
    is read from. Raises :class:`LeakageError` if the feature window would
    reach past the decision bar, or if the label does not actually lie in
    the future relative to it.
    """
    if window_end > decision_index:
        raise LeakageError(
            f"Feature window end ({window_end}) is after the decision index "
            f"({decision_index}) -- features would see the current/future bar early."
        )
    if window_start > window_end:
        raise LeakageError(f"Invalid window: start ({window_start}) > end ({window_end}).")
    if label_index <= decision_index:
        raise LeakageError(
            f"Label index ({label_index}) is not strictly after the decision index "
            f"({decision_index}) -- label would not describe the future."
        )


@dataclass(slots=True)
class ValidationReport:
    """Structured result of :func:`validate_dataset`, consumed by report generation."""

    n_samples: int = 0
    n_features: int = 0
    expected_n_features: int = 0
    feature_count_ok: bool = True
    duplicate_metadata_rows: int = 0
    nan_count: int = 0
    inf_count: int = 0
    target_alignment_ok: bool = True
    leakage_ok: bool = True
    leakage_violations: List[str] = field(default_factory=list)
    class_balance: Dict[str, int] = field(default_factory=dict)
    regression_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_dataset(dataset: "Dataset", expected_n_features: int | None = None) -> ValidationReport:
    """Run every post-build check against an assembled :class:`Dataset`."""
    report = ValidationReport()
    X = dataset.X
    n_samples, n_features = X.shape
    report.n_samples = n_samples
    report.n_features = n_features
    report.expected_n_features = expected_n_features or n_features
    report.feature_count_ok = n_features == report.expected_n_features
    if not report.feature_count_ok:
        report.errors.append(
            f"Feature count mismatch: got {n_features}, expected {report.expected_n_features}."
        )

    nan_mask = ~np.isfinite(X)
    report.nan_count = int(np.isnan(X).sum())
    report.inf_count = int(np.isinf(X).sum())
    if nan_mask.any():
        report.errors.append(
            f"{report.nan_count} NaN and {report.inf_count} Inf value(s) found in X after "
            "preprocessing -- the feature pipeline must leave X fully finite."
        )

    if len(dataset.metadata) != n_samples:
        report.target_alignment_ok = False
        report.errors.append(
            f"metadata has {len(dataset.metadata)} rows but X has {n_samples} samples."
        )
    for name, y in dataset.y_reg.items():
        if len(y) != n_samples:
            report.target_alignment_ok = False
            report.errors.append(f"y_reg[{name!r}] has {len(y)} rows but X has {n_samples}.")
    if dataset.y_cls is not None and len(dataset.y_cls) != n_samples:
        report.target_alignment_ok = False
        report.errors.append(f"y_cls has {len(dataset.y_cls)} rows but X has {n_samples}.")

    dup_count = int(dataset.metadata.duplicated(subset=["timestamp"]).sum())
    report.duplicate_metadata_rows = dup_count
    if dup_count:
        report.warnings.append(f"{dup_count} duplicate decision timestamp(s) in metadata.")

    if dataset.y_cls is not None and dataset.class_names:
        counts = np.bincount(dataset.y_cls, minlength=len(dataset.class_names))
        report.class_balance = {
            name: int(counts[i]) for i, name in enumerate(dataset.class_names)
        }

    for name, y in dataset.y_reg.items():
        finite = y[np.isfinite(y)]
        if finite.size:
            report.regression_stats[name] = {
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=0)),
                "min": float(finite.min()),
                "max": float(finite.max()),
                "median": float(np.median(finite)),
            }

    for j, name in enumerate(dataset.feature_names):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if finite.size:
            report.feature_stats[name] = {
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=0)),
                "min": float(finite.min()),
                "max": float(finite.max()),
            }

    return report
