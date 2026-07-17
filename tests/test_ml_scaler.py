"""Tests for ml_pipeline.scaler."""
from __future__ import annotations

import numpy as np
import pytest

from ml_pipeline.scaler import FeatureScaler


@pytest.fixture()
def X_train() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(loc=[10.0, -5.0, 100.0], scale=[2.0, 1.0, 20.0], size=(200, 3))


def test_standard_scaler_zero_mean_unit_variance(X_train: np.ndarray) -> None:
    scaler = FeatureScaler("standard")
    X_scaled = scaler.fit_transform(X_train)
    np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-9)


def test_minmax_scaler_bounds(X_train: np.ndarray) -> None:
    scaler = FeatureScaler("minmax")
    X_scaled = scaler.fit_transform(X_train)
    assert X_scaled.min() >= -1e-9
    assert X_scaled.max() <= 1 + 1e-9


def test_robust_scaler_runs(X_train: np.ndarray) -> None:
    scaler = FeatureScaler("robust")
    X_scaled = scaler.fit_transform(X_train)
    assert X_scaled.shape == X_train.shape
    assert np.isfinite(X_scaled).all()


def test_none_scaler_is_identity(X_train: np.ndarray) -> None:
    scaler = FeatureScaler("none")
    X_scaled = scaler.fit_transform(X_train)
    np.testing.assert_allclose(X_scaled, X_train)


def test_fit_only_on_train_not_refit_on_test(X_train: np.ndarray) -> None:
    scaler = FeatureScaler("standard")
    scaler.fit(X_train)
    X_test = X_train[:10] + 50.0  # shifted "test" data
    X_test_scaled = scaler.transform(X_test)
    # Using TRAIN stats, a +50 shift should NOT come out ~zero-mean.
    assert not np.allclose(X_test_scaled.mean(axis=0), 0.0, atol=1.0)


def test_transform_before_fit_raises(X_train: np.ndarray) -> None:
    scaler = FeatureScaler("standard")
    with pytest.raises(RuntimeError):
        scaler.transform(X_train)


def test_wrong_feature_count_raises(X_train: np.ndarray) -> None:
    scaler = FeatureScaler("standard").fit(X_train)
    with pytest.raises(ValueError):
        scaler.transform(X_train[:, :2])


def test_invalid_method_raises() -> None:
    with pytest.raises(ValueError):
        FeatureScaler("bogus")


def test_save_load_round_trip(tmp_path, X_train: np.ndarray) -> None:
    scaler = FeatureScaler("standard").fit(X_train)
    path = tmp_path / "scaler.joblib"
    scaler.save(path)
    loaded = FeatureScaler.load(path)
    np.testing.assert_allclose(loaded.transform(X_train), scaler.transform(X_train))
