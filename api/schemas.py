"""Request body / query param schemas for every API router."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MarketAnalyzeRequest(BaseModel):
    symbol: str = Field(default="EUR_USD", description="OANDA instrument, e.g. EUR_USD")
    timeframe: str = Field(default="M5", description="OANDA granularity, e.g. M5, M15, H1")
    window_size: int = Field(default=200, ge=50, description="Trailing candles fed to the Market Structure Engine")
    count: int = Field(default=250, ge=50, le=5000, description="Candles to fetch (must be >= window_size)")


class StrategyEvaluateRequest(BaseModel):
    symbol: str = Field(default="EUR_USD")
    timeframe: str = Field(default="M5")
    strategy_name: str = Field(default="trend_following", description="One of the registered Strategy Lab strategies -- see GET /strategies")
    window_size: int = Field(default=200, ge=50)
    count: int = Field(default=250, ge=50, le=5000)


class TrainRequest(BaseModel):
    symbol: str = Field(default="EUR_USD")
    timeframe: str = Field(default="M5")
    count: int = Field(default=2500, ge=300, le=5000, description="Historical candles to fetch for training")
    window_size: int = Field(default=200, ge=50, description="Trailing candles per Market Structure Engine window")
    horizon: int = Field(default=5, ge=1, description="Bars-ahead prediction horizon")
    stride: int = Field(default=3, ge=1, description="Step between consecutive training samples")


class PredictRequest(BaseModel):
    symbol: str = Field(default="EUR_USD")
    timeframe: str = Field(default="M5")
    count: int = Field(default=2500, ge=300, le=5000, description="Only used the first time this symbol/timeframe is requested (triggers training)")
    window_size: int = Field(default=200, ge=50)
    horizon: int = Field(default=5, ge=1)
    stride: int = Field(default=3, ge=1)


class DecisionRequest(BaseModel):
    symbol: str = Field(default="EUR_USD")
    timeframe: str = Field(default="M5")
    strategy_name: str = Field(default="trend_following", description="One of the registered Strategy Lab strategies -- see GET /strategies")
    count: int = Field(default=2500, ge=300, le=5000)
    window_size: int = Field(default=200, ge=50)
    horizon: int = Field(default=5, ge=1)
    stride: int = Field(default=3, ge=1)


class MonitorHealthRequest(BaseModel):
    symbol: str = Field(default="EUR_USD")
    timeframe: str = Field(default="M5")
    task_type: str = Field(default="regression", description="'regression' or 'classification'")
    count: int = Field(default=2500, ge=300, le=5000)
    window_size: int = Field(default=200, ge=50)
    horizon: int = Field(default=5, ge=1)
    stride: int = Field(default=3, ge=1)
