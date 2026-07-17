"""Tests for api.middleware.EngineVersionsMiddleware.

No FastAPI TestClient here -- this venv doesn't have the httpx dependency
Starlette's TestClient now requires, and this middleware doesn't need a
live request cycle to verify: it's a static header map read once at import
time, so checking that map directly against each package's own __version__
is a real, sufficient test.
"""
from __future__ import annotations

import decision_engine
import linear_regression
import logistic_regression
import market_structure
import ml_pipeline
import model_monitor
import strategy
import training

from api.main import app
from api.middleware import EngineVersionsMiddleware, _VERSION_HEADERS


def test_every_expected_engine_has_a_version_header() -> None:
    expected = {
        "X-Market-Structure-Version": market_structure.__version__,
        "X-Ml-Pipeline-Version": ml_pipeline.__version__,
        "X-Training-Version": training.__version__,
        "X-Strategy-Version": strategy.__version__,
        "X-Linear-Regression-Version": linear_regression.__version__,
        "X-Logistic-Regression-Version": logistic_regression.__version__,
        "X-Model-Monitor-Version": model_monitor.__version__,
        "X-Decision-Engine-Version": decision_engine.__version__,
    }
    assert _VERSION_HEADERS == expected


def test_header_values_are_strings_not_version_objects() -> None:
    assert all(isinstance(v, str) for v in _VERSION_HEADERS.values())


def test_middleware_is_registered_on_the_app() -> None:
    registered = [m.cls for m in app.user_middleware]
    assert EngineVersionsMiddleware in registered


def test_cors_middleware_remains_outermost() -> None:
    """CORS must wrap every other middleware (including this one) so
    preflight/error responses still get CORS headers -- Starlette builds
    the stack from user_middleware[::-1], so index 0 is outermost."""
    from fastapi.middleware.cors import CORSMiddleware

    registered = [m.cls for m in app.user_middleware]
    assert registered[0] is CORSMiddleware
