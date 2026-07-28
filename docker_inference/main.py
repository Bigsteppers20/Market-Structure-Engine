"""Entry point for the market_structure model container.

One container = one model, exposing its own /predict and /health directly
(no auto-discovery needed here -- that pattern is for a single process
serving multiple models; in this stack each model is already its own
container, and the gateway is what fans requests out across them).

Model loading (weights download + market_structure import) is kicked off
in a background thread from the lifespan startup hook, not at module
import time -- see inference.py's module docstring for why: it lets
uvicorn start accepting connections immediately, so /health can report a
real 503 ("still loading") instead of being unreachable while a slow
weights download is in progress. Railway's healthcheck (railway.json)
polls this endpoint and only routes traffic once it sees 200.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import inference
from .router import router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    inference.start_loading()
    yield


app = FastAPI(title="market_structure model", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
def health() -> JSONResponse:
    ready = inference.is_ready()
    payload = {
        "status": "ok" if ready else "not_ready",
        "ready": ready,
        "error": inference.load_error(),
        "mock_mode": inference.USE_MOCK_MODEL,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)
