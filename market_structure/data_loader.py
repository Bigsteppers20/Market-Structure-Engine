"""Data loading and validation.

Accepts a ``pandas.DataFrame`` or ``List[Candle]`` and returns a clean,
schema-validated, timestamp-sorted, de-duplicated OHLCV DataFrame ready for
the analysis engines.
"""
from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import pandas as pd

from .candle import Candle, candles_to_dataframe
from .config import EngineConfig

REQUIRED_COLUMNS: Sequence[str] = ("timestamp", "open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS: Sequence[str] = ("spread", "tick_volume")

InputData = Union[pd.DataFrame, List[Candle]]


class DataValidationError(ValueError):
    """Raised when input market data cannot be validated."""


class DataLoader:
    """Validates and normalizes raw OHLCV input.

    Parameters
    ----------
    config:
        Engine configuration; controls optional missing-candle filling.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def load(self, data: InputData) -> pd.DataFrame:
        """Validate and normalize input data.

        Steps: schema validation -> dtype normalization -> timestamp sort ->
        duplicate removal -> OHLC sanity checks -> optional gap filling.

        Returns a fresh DataFrame with a ``RangeIndex``.
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, list) and data and isinstance(data[0], Candle):
            df = candles_to_dataframe(data)
        elif isinstance(data, list) and not data:
            raise DataValidationError("Input candle list is empty.")
        else:
            raise DataValidationError(
                f"Unsupported input type: {type(data)!r}. "
                "Expected pandas.DataFrame or List[Candle]."
            )

        self._validate_schema(df)
        df = self._normalize_dtypes(df)
        df = df.sort_values("timestamp", kind="mergesort")
        df = df.drop_duplicates(subset="timestamp", keep="last")
        df = df.reset_index(drop=True)
        self._validate_values(df)
        if self.config.fill_missing_candles:
            df = self._fill_missing(df)
        return df

    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_schema(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")
        if len(df) < 3:
            raise DataValidationError("At least 3 candles are required.")

    @staticmethod
    def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        numeric = ["open", "high", "low", "close", "volume"]
        numeric += [c for c in OPTIONAL_COLUMNS if c in df.columns]
        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
        if df[["open", "high", "low", "close"]].isna().any().any():
            raise DataValidationError("OHLC columns contain non-numeric/NaN values.")
        df["volume"] = df["volume"].fillna(0.0)
        return df

    @staticmethod
    def _validate_values(df: pd.DataFrame) -> None:
        bad_hl = (df["high"] < df["low"]).any()
        if bad_hl:
            raise DataValidationError("Found candles where high < low.")
        bad_range = (
            (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
        ).any()
        if bad_range:
            raise DataValidationError("Found candles where open/close falls outside [low, high].")
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise DataValidationError("Non-positive prices detected.")

    @staticmethod
    def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
        """Reindex to the modal timestamp spacing and forward-fill gaps.

        Synthetic candles are flat (O=H=L=C=previous close) with zero volume.
        """
        deltas = df["timestamp"].diff().dropna()
        if deltas.empty:
            return df
        step = deltas.mode().iloc[0]
        if step <= pd.Timedelta(0):
            return df
        full_index = pd.date_range(df["timestamp"].iloc[0], df["timestamp"].iloc[-1], freq=step)
        df = df.set_index("timestamp").reindex(full_index)
        close_ff = df["close"].ffill()
        synthetic = df["open"].isna()
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].where(~synthetic, close_ff)
        df["volume"] = df["volume"].fillna(0.0)
        for col in OPTIONAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].ffill()
        df = df.rename_axis("timestamp").reset_index()
        return df
