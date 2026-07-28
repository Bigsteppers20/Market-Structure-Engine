"""Request/response contract for this container's POST /predict.

Mirrors the real input the Market Structure Engine's own ``DataLoader``
requires (required columns ``timestamp, open, high, low, close, volume``,
at least 3 candles) and the real output shape ``MarketState.to_dict()``
produces (a flat dict of ~185 named numeric features).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Candle(BaseModel):
    timestamp: str = Field(..., description="ISO-8601 timestamp, e.g. 2026-01-01T00:00:00Z")
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    spread: Optional[float] = None
    tick_volume: Optional[float] = None


class PredictRequest(BaseModel):
    candles: List[Candle] = Field(
        ..., min_length=3,
        description="OHLCV candles, any order (sorted internally by timestamp) -- at least 3 required.",
    )
    window_size: int = Field(
        default=200, ge=3,
        description="Only the most recent `window_size` candles (after sorting) are fed to the engine.",
    )
    swing_window: int = Field(
        default=5, ge=1, description="Swing-detection lookback bars (EngineConfig.swing_window upstream).",
    )


class PredictResponse(BaseModel):
    market_state: Dict[str, float] = Field(
        ..., description="Flat feature dict from the upstream engine's MarketState.to_dict() (or a mock, see USE_MOCK_MODEL).",
    )
    n_candles: int = Field(..., description="Candles actually analyzed after windowing.")
