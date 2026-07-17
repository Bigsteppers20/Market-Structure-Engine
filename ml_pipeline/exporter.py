"""Dataset export to CSV, Parquet, NumPy, joblib, and pickle."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .dataset_builder import Dataset


def _to_frame(dataset: "Dataset") -> pd.DataFrame:
    """Flatten a Dataset into one wide DataFrame: features + targets + metadata."""
    df = pd.DataFrame(dataset.X, columns=dataset.feature_names)
    for name, y in dataset.y_reg.items():
        df[f"target_reg_{name}"] = y
    if dataset.y_cls is not None:
        df["target_cls"] = dataset.y_cls
        df["target_cls_name"] = [dataset.class_names[i] for i in dataset.y_cls]
    meta = dataset.metadata.reset_index(drop=True)
    for col in meta.columns:
        df[f"meta_{col}"] = meta[col].to_numpy()
    return df


class DatasetExporter:
    """Static export helpers -- one method per output format."""

    @staticmethod
    def to_csv(dataset: "Dataset", path: str | Path) -> Path:
        path = Path(path)
        _to_frame(dataset).to_csv(path, index=False)
        return path

    @staticmethod
    def to_parquet(dataset: "Dataset", path: str | Path) -> Path:
        path = Path(path)
        _to_frame(dataset).to_parquet(path, index=False)
        return path

    @staticmethod
    def to_numpy(dataset: "Dataset", path: str | Path) -> Path:
        """Export as a compressed ``.npz`` archive (arrays only, no DataFrame)."""
        path = Path(path)
        arrays = {"X": dataset.X, "feature_names": np.array(dataset.feature_names, dtype=object)}
        for name, y in dataset.y_reg.items():
            arrays[f"y_reg_{name}"] = y
        if dataset.y_cls is not None:
            arrays["y_cls"] = dataset.y_cls
            arrays["class_names"] = np.array(dataset.class_names, dtype=object)
        arrays["timestamps"] = dataset.metadata["timestamp"].to_numpy()
        np.savez_compressed(path, **arrays)
        return path

    @staticmethod
    def to_joblib(dataset: "Dataset", path: str | Path) -> Path:
        path = Path(path)
        joblib.dump(dataset, path)
        return path

    @staticmethod
    def to_pickle(dataset: "Dataset", path: str | Path) -> Path:
        path = Path(path)
        with open(path, "wb") as fh:
            pickle.dump(dataset, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path
