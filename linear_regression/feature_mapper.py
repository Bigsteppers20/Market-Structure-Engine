"""MarketState -> numeric feature vector, and nothing else.

This is the *only* place in the engine that touches a ``MarketState``
object directly for the model's input side -- every other module works with
plain ``np.ndarray``/``List[str]``. Uses exclusively
``MarketState.to_vector()``/``to_dict()`` (the Market Structure Engine's own
public API); never reads a candle, computes an indicator, or inspects
``swings``/``breaks``/``chochs`` directly.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from market_structure import MarketState


def extract_feature_vector(market_state: MarketState) -> Tuple[np.ndarray, List[str]]:
    """Return ``(X, feature_names)`` with ``X`` shaped ``(1, n_features)``,
    ready to feed into a fitted preprocessing pipeline + model."""
    vector, names = market_state.to_vector()
    return vector.reshape(1, -1), names


def feature_completeness(market_state: MarketState) -> float:
    """Fraction (0-1) of the engine's own ``_valid`` flags that are true.

    A direct, engine-native proxy for "how ready is this MarketState" --
    used by the confidence engine's feature-completeness factor. Reads only
    ``MarketState.to_dict()``.
    """
    state = market_state.to_dict()
    valid_keys = [k for k in state if k.endswith("_valid")]
    if not valid_keys:
        return 1.0
    return sum(state[k] for k in valid_keys) / len(valid_keys)
