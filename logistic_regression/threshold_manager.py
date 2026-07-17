"""Per-class decision thresholds, as an alternative to plain argmax.

Useful when the default argmax decision rule under- or over-fires a
minority class (e.g. NO_TRADE dominating a heavily imbalanced dataset);
``optimize()`` fits one threshold per class (one-vs-rest, maximizing F1 via
grid search) on labeled probability data, and ``apply()`` uses those
thresholds instead of a bare argmax at prediction time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score

from .probability_engine import predicted_class

THRESHOLD_STRATEGIES = ("argmax", "optimized", "custom")


@dataclass(slots=True)
class ThresholdManager:
    """Holds and applies per-class decision thresholds."""

    classes: List[str]
    strategy: str = "argmax"
    thresholds: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in THRESHOLD_STRATEGIES:
            raise ValueError(f"strategy={self.strategy!r}, expected one of {THRESHOLD_STRATEGIES}.")

    def optimize(self, y_true_encoded: np.ndarray, probabilities: np.ndarray, grid_size: int = 19) -> Dict[str, float]:
        """Grid-search a per-class threshold maximizing one-vs-rest F1."""
        y_true_encoded = np.asarray(y_true_encoded)
        thresholds: Dict[str, float] = {}
        for i, cls in enumerate(self.classes):
            y_binary = (y_true_encoded == i).astype(int)
            best_threshold, best_f1 = 0.5, -1.0
            for t in np.linspace(0.05, 0.95, grid_size):
                preds = (probabilities[:, i] >= t).astype(int)
                score = f1_score(y_binary, preds, zero_division=0)
                if score > best_f1:
                    best_f1, best_threshold = score, float(t)
            thresholds[cls] = best_threshold
        self.thresholds = thresholds
        return thresholds

    def apply(self, probabilities: Dict[str, float]) -> str:
        """Decide a class using the configured strategy."""
        if self.strategy == "argmax" or not self.thresholds:
            return predicted_class(probabilities)
        candidates = {c: p for c, p in probabilities.items() if p >= self.thresholds.get(c, 0.0)}
        if candidates:
            return predicted_class(candidates)
        return predicted_class(probabilities)  # nothing clears its threshold -- fall back to argmax
