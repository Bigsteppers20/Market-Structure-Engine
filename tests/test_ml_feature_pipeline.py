"""Tests for ml_pipeline.feature_pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from ml_pipeline.config import DatasetConfig
from ml_pipeline.feature_pipeline import FeaturePipeline


def _names() -> list[str]:
    return ["ind_atr", "ind_atr_valid", "ind_rsi", "ind_rsi_valid", "session_hour", "trend_direction"]


def test_impute_invalid_uses_train_median() -> None:
    names = _names()
    # rows: atr values 1,2,3 all valid; row 4 atr=999 but INVALID -> should be imputed
    X = np.array([
        [1.0, 1.0, 50.0, 1.0, 10.0, 1.0],
        [2.0, 1.0, 55.0, 1.0, 11.0, 0.0],
        [3.0, 1.0, 60.0, 1.0, 12.0, -1.0],
        [999.0, 0.0, 999.0, 0.0, 13.0, 1.0],
    ], dtype=np.float64)
    cfg = DatasetConfig(impute_invalid=True)
    fp = FeaturePipeline(cfg)
    X_out, names_out = fp.fit_transform(X, names)
    assert names_out == names
    assert X_out[3, 0] == pytest.approx(2.0)  # median of [1,2,3]
    assert X_out[3, 2] == pytest.approx(55.0)  # median of [50,55,60]
    # valid flags themselves are untouched
    assert X_out[3, 1] == 0.0
    assert X_out[3, 3] == 0.0


def test_impute_disabled_leaves_raw_values() -> None:
    names = _names()
    X = np.array([[1.0, 1.0, 50.0, 1.0, 10.0, 1.0], [999.0, 0.0, 999.0, 0.0, 13.0, 1.0]])
    cfg = DatasetConfig(impute_invalid=False)
    fp = FeaturePipeline(cfg)
    X_out, _ = fp.fit_transform(X, names)
    assert X_out[1, 0] == 999.0


def test_reorder_on_transform() -> None:
    names = _names()
    X = np.arange(len(names), dtype=np.float64).reshape(1, -1)
    cfg = DatasetConfig(impute_invalid=False)
    fp = FeaturePipeline(cfg)
    fp.fit(X, names)
    shuffled_names = list(reversed(names))
    X_shuffled = X[:, ::-1]
    X_out, out_names = fp.transform(X_shuffled, shuffled_names)
    assert out_names == names
    np.testing.assert_allclose(X_out, X)


def test_transform_before_fit_raises() -> None:
    fp = FeaturePipeline(DatasetConfig())
    with pytest.raises(RuntimeError):
        fp.transform(np.zeros((1, 3)), ["a", "b", "c"])


def test_missing_feature_at_transform_raises() -> None:
    names = _names()
    X = np.zeros((2, len(names)))
    fp = FeaturePipeline(DatasetConfig(impute_invalid=False))
    fp.fit(X, names)
    with pytest.raises(ValueError):
        fp.transform(X[:, :-1], names[:-1])


def test_cyclical_encoding_replaces_hour_with_sin_cos() -> None:
    names = _names()
    X = np.array([[1.0, 1.0, 50.0, 1.0, 0.0, 1.0], [1.0, 1.0, 50.0, 1.0, 12.0, 1.0]])
    cfg = DatasetConfig(impute_invalid=False, cyclical_time_encoding=True)
    fp = FeaturePipeline(cfg)
    X_out, names_out = fp.fit_transform(X, names)
    assert "session_hour" not in names_out
    assert "session_hour_sin" in names_out and "session_hour_cos" in names_out
    sin_idx = names_out.index("session_hour_sin")
    cos_idx = names_out.index("session_hour_cos")
    # hour=0 -> sin=0, cos=1
    assert X_out[0, sin_idx] == pytest.approx(0.0, abs=1e-9)
    assert X_out[0, cos_idx] == pytest.approx(1.0, abs=1e-9)
    # hour=12 (half period of 24) -> sin=0, cos=-1
    assert X_out[1, sin_idx] == pytest.approx(0.0, abs=1e-9)
    assert X_out[1, cos_idx] == pytest.approx(-1.0, abs=1e-9)


def test_one_hot_categorical_expands_direction_code() -> None:
    names = _names()
    X = np.array([
        [1.0, 1.0, 50.0, 1.0, 5.0, -1.0],
        [1.0, 1.0, 50.0, 1.0, 5.0, 0.0],
        [1.0, 1.0, 50.0, 1.0, 5.0, 1.0],
    ])
    cfg = DatasetConfig(impute_invalid=False, one_hot_categorical=True)
    fp = FeaturePipeline(cfg)
    X_out, names_out = fp.fit_transform(X, names)
    assert "trend_direction" not in names_out
    for suffix in ("neg", "zero", "pos"):
        assert f"trend_direction_{suffix}" in names_out
    neg_idx = names_out.index("trend_direction_neg")
    zero_idx = names_out.index("trend_direction_zero")
    pos_idx = names_out.index("trend_direction_pos")
    np.testing.assert_array_equal(X_out[:, neg_idx], [1, 0, 0])
    np.testing.assert_array_equal(X_out[:, zero_idx], [0, 1, 0])
    np.testing.assert_array_equal(X_out[:, pos_idx], [0, 0, 1])
