"""Tests for linear_regression.feature_mapper."""
from __future__ import annotations

import numpy as np

from linear_regression.feature_mapper import extract_feature_vector, feature_completeness


def test_extract_feature_vector_shape_matches_engine(market_state) -> None:
    X, names = extract_feature_vector(market_state)
    vector, expected_names = market_state.to_vector()
    assert X.shape == (1, vector.shape[0])
    assert names == expected_names
    np.testing.assert_allclose(X.ravel(), vector)


def test_feature_completeness_in_unit_interval(market_state) -> None:
    completeness = feature_completeness(market_state)
    assert 0.0 <= completeness <= 1.0


def test_feature_completeness_lower_on_short_history() -> None:
    from conftest import make_ohlcv
    from market_structure import EngineConfig, MarketStructureEngine

    short_df = make_ohlcv(30, seed=2)
    engine = MarketStructureEngine(EngineConfig(swing_window=3))
    engine.load(short_df)
    engine.analyze()
    short_state = engine.market_state()

    long_df = make_ohlcv(1000, seed=2)
    engine2 = MarketStructureEngine(EngineConfig(swing_window=3))
    engine2.load(long_df)
    engine2.analyze()
    long_state = engine2.market_state()

    assert feature_completeness(short_state) <= feature_completeness(long_state)
