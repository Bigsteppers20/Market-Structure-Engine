"""Tests for ml_pipeline.exporter -- round-trip every export format."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.dataset_builder import Dataset
from ml_pipeline.exporter import DatasetExporter


@pytest.fixture()
def dataset() -> Dataset:
    n = 8
    X = np.arange(n * 4, dtype=np.float64).reshape(n, 4)
    names = ["f0", "f1", "f2", "f3"]
    y_reg = {"next_close": np.linspace(1.0, 2.0, n)}
    y_cls = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    meta = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="5min"),
        "symbol": "EUR_USD", "timeframe": "M5",
        "window_start": pd.date_range("2025-01-01", periods=n, freq="5min"),
        "window_end": pd.date_range("2025-01-01", periods=n, freq="5min"),
    })
    return Dataset(X=X, feature_names=names, y_reg=y_reg, y_cls=y_cls,
                    class_names=["SELL", "NO_TRADE", "BUY"], metadata=meta)


def test_to_csv_round_trip(tmp_path, dataset: Dataset) -> None:
    path = DatasetExporter.to_csv(dataset, tmp_path / "d.csv")
    assert path.exists() and path.stat().st_size > 0
    df = pd.read_csv(path)
    assert len(df) == len(dataset)
    assert "target_reg_next_close" in df.columns
    assert "target_cls" in df.columns


def test_to_parquet_round_trip(tmp_path, dataset: Dataset) -> None:
    path = DatasetExporter.to_parquet(dataset, tmp_path / "d.parquet")
    df = pd.read_parquet(path)
    assert len(df) == len(dataset)


def test_to_numpy_round_trip(tmp_path, dataset: Dataset) -> None:
    path = DatasetExporter.to_numpy(dataset, tmp_path / "d.npz")
    loaded = np.load(path, allow_pickle=True)
    np.testing.assert_allclose(loaded["X"], dataset.X)
    np.testing.assert_array_equal(loaded["y_cls"], dataset.y_cls)


def test_to_joblib_round_trip(tmp_path, dataset: Dataset) -> None:
    path = DatasetExporter.to_joblib(dataset, tmp_path / "d.joblib")
    import joblib
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.X, dataset.X)


def test_to_pickle_round_trip(tmp_path, dataset: Dataset) -> None:
    path = DatasetExporter.to_pickle(dataset, tmp_path / "d.pkl")
    import pickle
    with open(path, "rb") as fh:
        loaded = pickle.load(fh)
    np.testing.assert_allclose(loaded.X, dataset.X)
    assert loaded.feature_names == dataset.feature_names
