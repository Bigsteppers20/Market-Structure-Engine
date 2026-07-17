"""Probability vector mechanics: normalization, predicted class, margin,
and entropy -- pure functions over a ``{class_name: probability}`` mapping.

Distinct from ``confidence.py`` (a broader multi-factor 0-100 trust score)
and ``calibration.py`` (transforms raw scores into calibrated
probabilities): this module only manipulates the *already-produced*
probability vector.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def normalize_probabilities(raw: np.ndarray) -> np.ndarray:
    """Force exact sum-to-1 per row (guards against floating-point drift)."""
    raw = np.clip(np.asarray(raw, dtype=float), 1e-12, None)
    return raw / raw.sum(axis=-1, keepdims=True)


def to_class_probabilities(proba_row: np.ndarray, classes: List[str]) -> Dict[str, float]:
    probs = normalize_probabilities(proba_row)
    return {cls: float(p) for cls, p in zip(classes, probs)}


def predicted_class(probabilities: Dict[str, float]) -> str:
    return max(probabilities, key=probabilities.get)


def probability_margin(probabilities: Dict[str, float]) -> float:
    """Gap between the top-1 and top-2 probabilities -- 0 = maximally
    ambiguous between the two leading classes, close to 1 = decisive."""
    ordered = sorted(probabilities.values(), reverse=True)
    if len(ordered) < 2:
        return ordered[0] if ordered else 0.0
    return float(ordered[0] - ordered[1])


def prediction_entropy(probabilities: Dict[str, float]) -> float:
    """Normalized Shannon entropy in [0, 1]: 0 = fully certain (one class at
    probability 1), 1 = maximally uncertain (uniform over all classes)."""
    p = np.array(list(probabilities.values()), dtype=float)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    raw_entropy = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(len(probabilities)))
    return raw_entropy / max_entropy if max_entropy > 0 else 0.0


def assert_probabilities_sum_to_one(probabilities: Dict[str, float], atol: float = 1e-9) -> None:
    total = sum(probabilities.values())
    if abs(total - 1.0) > atol:
        raise ValueError(f"Probabilities sum to {total!r}, expected 1.0 (+/-{atol}). Got: {probabilities}")
