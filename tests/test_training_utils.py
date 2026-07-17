"""Tests for training.utils."""
from __future__ import annotations

import time

from training.utils import Timer, get_git_commit, hash_strings, new_id, set_random_seed, utc_timestamp


def test_new_id_is_unique_and_prefixed() -> None:
    a, b = new_id("exp"), new_id("exp")
    assert a != b
    assert a.startswith("exp_") and b.startswith("exp_")


def test_utc_timestamp_is_iso_format() -> None:
    ts = utc_timestamp()
    assert "T" in ts
    assert ts.count("-") >= 2


def test_hash_strings_stable_and_order_sensitive() -> None:
    h1 = hash_strings(["a", "b", "c"])
    h2 = hash_strings(["a", "b", "c"])
    h3 = hash_strings(["c", "b", "a"])
    assert h1 == h2
    assert h1 != h3


def test_get_git_commit_never_raises(tmp_path) -> None:
    # tmp_path is not a git repo -- must return None, not raise.
    commit = get_git_commit(tmp_path)
    assert commit is None


def test_timer_measures_elapsed() -> None:
    with Timer() as t:
        time.sleep(0.01)
    assert t.elapsed_seconds >= 0.01


def test_set_random_seed_reproducible() -> None:
    import numpy as np
    set_random_seed(123)
    a = np.random.rand(5)
    set_random_seed(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)
