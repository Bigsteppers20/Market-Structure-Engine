"""Time-series-aware train/validation/test splitting.

Every method here produces index ranges by **chronological position only**
-- there is no shuffling anywhere in this module, because shuffling a time
series before splitting leaks future information into the training set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import numpy as np


@dataclass(slots=True)
class SplitResult:
    """One chronological split: index arrays into the dataset, in time order."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def __post_init__(self) -> None:
        all_idx = np.concatenate([self.train_idx, self.val_idx, self.test_idx])
        if len(set(all_idx.tolist())) != len(all_idx):
            raise ValueError("SplitResult contains overlapping indices between splits.")
        if self.train_idx.size and self.val_idx.size and self.train_idx.max() >= self.val_idx.min():
            raise ValueError("train indices must all precede val indices chronologically.")
        if self.val_idx.size and self.test_idx.size and self.val_idx.max() >= self.test_idx.min():
            raise ValueError("val indices must all precede test indices chronologically.")
        if (not self.val_idx.size) and self.train_idx.size and self.test_idx.size:
            if self.train_idx.max() >= self.test_idx.min():
                raise ValueError("train indices must all precede test indices chronologically.")


class TimeSeriesSplitter:
    """Chronological splitting -- simple 3-way split or walk-forward folds.

    Parameters
    ----------
    method:
        ``"simple"`` -- one chronological train/val/test split by fraction.
        ``"walk_forward"`` -- multiple folds, each fold's test block
        immediately follows its train block (see ``window_mode``).
    window_mode:
        For ``"walk_forward"`` only: ``"expanding"`` (train grows to include
        all prior folds) or ``"rolling"`` (train is a fixed-size sliding
        window, oldest data dropped as new folds are added).
    """

    def __init__(
        self,
        method: str = "simple",
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        window_mode: str = "expanding",
        n_folds: int = 5,
        train_size: int | None = None,
        test_size: int | None = None,
    ) -> None:
        if method not in ("simple", "walk_forward"):
            raise ValueError("method must be 'simple' or 'walk_forward'.")
        if window_mode not in ("expanding", "rolling"):
            raise ValueError("window_mode must be 'expanding' or 'rolling'.")
        if not (0 < train_frac < 1) or not (0 <= val_frac < 1) or train_frac + val_frac >= 1:
            raise ValueError("Require 0 < train_frac < 1, 0 <= val_frac < 1, train_frac+val_frac < 1.")
        self.method = method
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.window_mode = window_mode
        self.n_folds = n_folds
        self.train_size = train_size
        self.test_size = test_size

    def split(self, n_samples: int) -> List[SplitResult]:
        """Return one or more :class:`SplitResult`, in chronological order.

        ``"simple"`` always returns exactly one result. ``"walk_forward"``
        returns ``n_folds`` results (each with an empty ``val_idx`` -- use
        each fold's ``train_idx``/``test_idx`` directly for walk-forward
        validation).
        """
        if n_samples < 3:
            raise ValueError("Need at least 3 samples to split.")
        if self.method == "simple":
            return [self._simple_split(n_samples)]
        return list(self._walk_forward_split(n_samples))

    # ------------------------------------------------------------------ #
    def _simple_split(self, n_samples: int) -> SplitResult:
        n_train = int(n_samples * self.train_frac)
        n_val = int(n_samples * self.val_frac)
        n_train = max(n_train, 1)
        n_val = max(n_val, 0) if self.val_frac > 0 else 0
        if n_train + n_val >= n_samples:
            n_val = max(n_samples - n_train - 1, 0)
        idx = np.arange(n_samples)
        return SplitResult(
            train_idx=idx[:n_train],
            val_idx=idx[n_train: n_train + n_val],
            test_idx=idx[n_train + n_val:],
        )

    def _walk_forward_split(self, n_samples: int) -> Iterator[SplitResult]:
        n_folds = self.n_folds
        test_size = self.test_size or max(n_samples // (n_folds + 1), 1)
        min_train = self.train_size or test_size
        idx = np.arange(n_samples)

        first_test_start = min_train
        max_folds = max((n_samples - first_test_start) // test_size, 0)
        folds = min(n_folds, max_folds)
        if folds < 1:
            raise ValueError(
                f"Cannot form any walk-forward fold from {n_samples} samples with "
                f"train_size>={min_train} and test_size={test_size}."
            )

        for fold in range(folds):
            test_start = first_test_start + fold * test_size
            test_end = min(test_start + test_size, n_samples)
            if self.window_mode == "expanding":
                train_start = 0
            else:  # rolling: fixed-size window immediately preceding the test block
                train_start = max(0, test_start - min_train)
            yield SplitResult(
                train_idx=idx[train_start:test_start],
                val_idx=np.array([], dtype=int),
                test_idx=idx[test_start:test_end],
            )
