"""Market Structure Engine endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter(prefix="/market", tags=["Market Structure Engine"])


@router.get("/analyze")
def analyze(
    request: Request,
    symbol: str = Query(default="EUR_USD", description="OANDA instrument, e.g. EUR_USD"),
    timeframe: str = Query(default="M5", description="OANDA granularity, e.g. M5, M15, H1"),
    window_size: int = Query(default=200, ge=50, description="Trailing candles fed to the Market Structure Engine"),
    count: int = Query(default=250, ge=50, le=5000, description="Candles to fetch (must be >= window_size)"),
) -> dict:
    """Fetch live OANDA candles and return the full ``MarketState`` (185-dim
    feature dict) -- no strategy, regression, or classification layered on top.
    Built once by ``MarketStateMiddleware`` from these same query params and
    attached to ``request.state.market_state`` before this handler runs.

    Example: ``GET /market/analyze?symbol=EUR_USD&timeframe=M5``
    """
    return request.state.market_state.to_dict()
