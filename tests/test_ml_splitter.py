"""Tests for ml_pipeline.splitter -- chronological, never-shuffled splitting."""
from __future__ import annotations

import numpy as np
import pytest

from ml_pipeline.splitter import SplitResult, TimeSeriesSplitter


def test_simple_split_chronological_and_no_overlap() -> None:
    splitter = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15)
    result = splitter.split(100)[0]
    assert len(result.train_idx) + len(result.val_idx) + len(result.test_idx) == 100
    assert result.train_idx.max() < result.val_idx.min()
    assert result.val_idx.max() < result.test_idx.min()
    # strictly increasing within each split (no shuffling)
    assert np.all(np.diff(result.train_idx) == 1)


def test_simple_split_no_validation_set() -> None:
    splitter = TimeSeriesSplitter(method="simple", train_frac=0.8, val_frac=0.0)
    result = splitter.split(50)[0]
    assert len(result.val_idx) == 0
    assert result.train_idx.max() < result.test_idx.min()


def test_split_result_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        SplitResult(train_idx=np.array([0, 1, 2]), val_idx=np.array([2, 3]), test_idx=np.array([4]))


def test_split_result_rejects_out_of_order_splits() -> None:
    with pytest.raises(ValueError):
        SplitResult(train_idx=np.array([5, 6, 7]), val_idx=np.array([0, 1]), test_idx=np.array([8]))


def test_walk_forward_expanding_train_grows() -> None:
    splitter = TimeSeriesSplitter(method="walk_forward", window_mode="expanding", n_folds=3, test_size=20)
    folds = splitter.split(120)
    assert len(folds) == 3
    for i in range(1, len(folds)):
        assert len(folds[i].train_idx) > len(folds[i - 1].train_idx)
        assert folds[i].train_idx.min() == 0  # expanding always starts at 0


def test_walk_forward_rolling_train_size_fixed() -> None:
    splitter = TimeSeriesSplitter(
        method="walk_forward", window_mode="rolling", n_folds=3, test_size=20, train_size=30
    )
    folds = splitter.split(150)
    sizes = {len(f.train_idx) for f in folds}
    assert sizes == {30}
    # rolling window should shift forward each fold, not always start at 0
    starts = [f.train_idx.min() for f in folds]
    assert starts == sorted(starts)
    assert starts[-1] > starts[0]


def test_walk_forward_folds_never_overlap_train_and_test() -> None:
    splitter = TimeSeriesSplitter(method="walk_forward", window_mode="expanding", n_folds=4, test_size=15)
    for fold in splitter.split(200):
        assert fold.train_idx.max() < fold.test_idx.min()


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        TimeSeriesSplitter(method="bogus")
    with pytest.raises(ValueError):
        TimeSeriesSplitter(window_mode="bogus")
    with pytest.raises(ValueError):
        TimeSeriesSplitter(train_frac=0.9, val_frac=0.2)  # sums >= 1


def test_too_few_samples_raises() -> None:
    with pytest.raises(ValueError):
        TimeSeriesSplitter().split(2)
