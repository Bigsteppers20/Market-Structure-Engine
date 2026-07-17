"""Rolling-window dataset construction on top of the Market Structure Engine.

For every decision candle ``t`` (subject to ``stride``), :class:`DatasetBuilder`:

1. Slices the trailing ``window_size`` candles ending at ``t`` (never later).
2. Runs a fresh :class:`~market_structure.MarketStructureEngine` pass over
   that window to produce one 185-dimensional feature vector.
3. Computes regression/classification labels from candles strictly *after*
   ``t`` (``t + horizon``).
4. Asserts, per sample, that the feature window and the label never overlap
   (:func:`~ml_pipeline.validator.assert_no_future_leakage`).

The Market Structure Engine itself is never modified or subclassed -- this
module only calls its public ``load()`` / ``analyze()`` / ``feature_vector()``
API, exactly as documented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from market_structure import MarketStructureEngine
from market_structure.candle import Candle, candles_to_dataframe

from .config import DatasetConfig
from .label_generator import (
    LabelGenerator,
    build_classification_label_generator,
    compute_regression_targets,
)
from .validator import (
    ValidationReport,
    assert_no_future_leakage,
    validate_dataset,
    validate_input_data,
)

InputData = Union[pd.DataFrame, str, Path, List[Candle]]


@dataclass(slots=True)
class Dataset:
    """A supervised-learning-ready dataset built from historical OHLCV data.

    Attributes
    ----------
    X:
        Feature matrix, shape ``(n_samples, n_features)``, float64. Exactly
        the Market Structure Engine's own ``feature_vector()`` output for
        each sample's window -- no scaling, imputation, or selection applied
        (that is :mod:`feature_pipeline`, :mod:`scaler`, and
        :mod:`feature_selector`'s job).
    feature_names:
        Column names for ``X``, in order (matches the engine's own ordering).
    y_reg:
        ``target_name -> (n_samples,)`` float64 array, one entry per
        configured regression target.
    y_cls:
        ``(n_samples,)`` int64 array of classification labels, or ``None``
        if no classification label generator was used.
    class_names:
        Ordered class names; ``y_cls[i] == j`` means class ``class_names[j]``.
    metadata:
        One row per sample: ``timestamp`` (decision candle), ``symbol``,
        ``timeframe``, ``window_start``, ``window_end``.
    """

    X: np.ndarray
    feature_names: List[str]
    y_reg: Dict[str, np.ndarray]
    y_cls: Optional[np.ndarray]
    class_names: List[str]
    metadata: pd.DataFrame

    def __len__(self) -> int:
        return self.X.shape[0]


class DatasetBuilder:
    """Builds a :class:`Dataset` from historical OHLCV data via a leakage-safe
    rolling window over the Market Structure Engine."""

    def __init__(self, config: DatasetConfig, label_generator: Optional[LabelGenerator] = None) -> None:
        self.config = config
        self.label_generator = label_generator or build_classification_label_generator(
            config.classification_label, **config.classification_kwargs
        )
        self._engine = MarketStructureEngine(config.engine_config)
        self.last_input_issues: List[str] = []
        self.last_report: Optional[ValidationReport] = None

    # ------------------------------------------------------------------ #
    def build(self, data: InputData) -> Dataset:
        """Run the full rolling-window build and return a validated :class:`Dataset`."""
        df = self._load_input(data)
        self.last_input_issues = validate_input_data(df)

        cfg = self.config
        n = len(df)
        start = cfg.window_size - 1
        stop = n - cfg.horizon
        if stop <= start:
            raise ValueError(
                f"Not enough candles ({n}) for window_size={cfg.window_size} and "
                f"horizon={cfg.horizon} -- need at least {cfg.window_size + cfg.horizon}."
            )

        rows_x: List[np.ndarray] = []
        feature_names: Optional[List[str]] = None
        y_reg_lists: Dict[str, List[float]] = {t: [] for t in cfg.regression_targets}
        y_cls_labels: List[str] = []
        meta_rows: List[dict] = []

        timestamps = df["timestamp"]
        for t in range(start, stop, cfg.stride):
            window_start = t - cfg.window_size + 1
            window_end = t
            label_index = t + cfg.horizon
            assert_no_future_leakage(t, window_start, window_end, label_index)

            window = df.iloc[window_start: window_end + 1]
            self._engine.load(window)
            self._engine.analyze()
            vec, names = self._engine.feature_vector()
            if feature_names is None:
                feature_names = names
            rows_x.append(vec)

            targets = compute_regression_targets(
                df, t, cfg.horizon, cfg.regression_targets, pip_size=cfg.pip_size
            )
            for tname in cfg.regression_targets:
                y_reg_lists[tname].append(targets[tname])

            y_cls_labels.append(self.label_generator.label(df, t, cfg.horizon))

            meta_rows.append({
                "timestamp": timestamps.iloc[t],
                "symbol": cfg.symbol,
                "timeframe": cfg.timeframe,
                "window_start": timestamps.iloc[window_start],
                "window_end": timestamps.iloc[window_end],
            })

        assert feature_names is not None
        X = np.vstack(rows_x)
        y_reg = {k: np.asarray(v, dtype=np.float64) for k, v in y_reg_lists.items()}
        class_names = list(self.label_generator.classes)
        class_to_int = {c: i for i, c in enumerate(class_names)}
        y_cls = np.asarray([class_to_int[c] for c in y_cls_labels], dtype=np.int64)
        metadata = pd.DataFrame(meta_rows)

        dataset = Dataset(
            X=X, feature_names=list(feature_names), y_reg=y_reg, y_cls=y_cls,
            class_names=class_names, metadata=metadata,
        )

        report = validate_dataset(dataset, expected_n_features=X.shape[1])
        report.leakage_ok = True  # every sample above passed assert_no_future_leakage or build() would have raised
        if self.last_input_issues:
            report.warnings.extend(self.last_input_issues)
        self.last_report = report
        return dataset

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_input(data: InputData) -> pd.DataFrame:
        """Normalize any accepted input type into a plain OHLCV DataFrame.

        Schema/dtype validation itself is deferred to the Market Structure
        Engine's own ``DataLoader`` (invoked per-window in ``build()``) so
        validation logic is never duplicated.
        """
        if isinstance(data, pd.DataFrame):
            return data.reset_index(drop=True)
        if isinstance(data, list):
            if data and isinstance(data[0], Candle):
                return candles_to_dataframe(data)
            raise TypeError("List input must be a List[Candle].")
        if isinstance(data, (str, Path)):
            path = Path(data)
            suffix = path.suffix.lower()
            if suffix == ".csv":
                return pd.read_csv(path)
            if suffix in (".parquet", ".pq"):
                return pd.read_parquet(path)
            raise ValueError(f"Unsupported file extension: {suffix!r} (expected .csv or .parquet)")
        raise TypeError(
            f"Unsupported input type: {type(data)!r}. "
            "Expected pandas.DataFrame, List[Candle], or a .csv/.parquet path."
        )
