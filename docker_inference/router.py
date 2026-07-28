"""Public HTTP contract for the Market Structure model container. Kept
thin by design: validation (via ``schemas.py``) + delegation to
``inference.run()`` only. All model logic lives in ``inference.py``.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from . import inference
from .schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not inference.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Market Structure model is not ready (failed to load at startup).",
        )
    try:
        result = inference.run(request.model_dump())
    except Exception:
        # Never leak the real exception/stack trace to the client -- log it
        # server-side and return a safe, generic message instead.
        logger.exception("Market Structure inference failed")
        raise HTTPException(status_code=500, detail="Inference failed.") from None
    return PredictResponse(**result)
