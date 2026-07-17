"""Label generation for supervised learning targets.

Two distinct kinds of label live here, both intentionally forward-looking
(that is the entire point of a supervised-learning target -- see the module
docstring note on leakage below):

* **Regression targets** -- a fixed registry of named continuous targets
  (:data:`REGRESSION_REGISTRY`), each a pure function of
  ``(df, index, horizon)``.
* **Classification labels** -- a pluggable :class:`LabelGenerator` interface
  so callers can swap in their own labeling rule without touching the
  dataset builder. :class:`ThresholdLabelGenerator` is the default
  (BUY / SELL / NO_TRADE from a forward-return threshold).

Leakage note
------------
Every function here reads ``df.iloc[index + 1 : index + 1 + horizon]`` (or
``df.iloc[index + horizon]``) quite deliberately -- a label describing "what
happens next" must, by definition, look at what happens next. The leakage
boundary this whole package protects is the **feature window** (see
``dataset_builder.py``), which must never read past ``index``. Labels and
features are computed from clearly separated slices of ``df``, and
``validator.py`` asserts this separation holds for every sample.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Dict, List, Sequence, Tuple, Type

import numpy as np
import pandas as pd

RegressionTargetFn = Callable[[pd.DataFrame, int, int, float], float]


def _close(df: pd.DataFrame, i: int) -> float:
    return float(df["close"].iloc[i])


def _next_close(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    return float(df["close"].iloc[index + horizon])


def _next_open(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    return float(df["open"].iloc[index + horizon])


def _next_high(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    return float(df["high"].iloc[index + horizon])


def _next_low(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    return float(df["low"].iloc[index + horizon])


def _next_return(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    c0 = _close(df, index)
    c1 = _close(df, index + horizon)
    return (c1 - c0) / c0 if c0 else 0.0


def _next_log_return(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    c0 = _close(df, index)
    c1 = _close(df, index + horizon)
    if c0 <= 0 or c1 <= 0:
        return 0.0
    return float(np.log(c1 / c0))


def _expected_pip_movement(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    c0 = _close(df, index)
    c1 = _close(df, index + horizon)
    return (c1 - c0) / pip_size if pip_size else 0.0


def _expected_percentage_change(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    return _next_return(df, index, horizon, pip_size) * 100.0


def _future_atr(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    """Mean True Range over the ``horizon`` bars strictly after ``index``."""
    end = min(index + 1 + horizon, len(df))
    seg = df.iloc[index + 1: end]
    if seg.empty:
        return 0.0
    high = seg["high"].to_numpy(dtype=float)
    low = seg["low"].to_numpy(dtype=float)
    close = seg["close"].to_numpy(dtype=float)
    prev_close = np.concatenate(([_close(df, index)], close[:-1]))
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    return float(tr.mean())


def _future_volatility(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    """Stdev of log-returns over the ``horizon`` bars strictly after ``index``."""
    end = min(index + 1 + horizon, len(df))
    closes = df["close"].iloc[index: end].to_numpy(dtype=float)
    if closes.size < 2:
        return 0.0
    rets = np.diff(np.log(np.maximum(closes, 1e-12)))
    return float(np.std(rets, ddof=0))


REGRESSION_REGISTRY: Dict[str, RegressionTargetFn] = {
    "next_close": _next_close,
    "next_open": _next_open,
    "next_high": _next_high,
    "next_low": _next_low,
    "next_return": _next_return,
    "next_log_return": _next_log_return,
    "expected_pip_movement": _expected_pip_movement,
    "expected_percentage_change": _expected_percentage_change,
    "future_atr": _future_atr,
    "future_volatility": _future_volatility,
}


def compute_regression_targets(
    df: pd.DataFrame, index: int, horizon: int, targets: Sequence[str], pip_size: float = 0.0001
) -> Dict[str, float]:
    """Compute every requested regression target for one decision candle."""
    return {name: REGRESSION_REGISTRY[name](df, index, horizon, pip_size) for name in targets}


# --------------------------------------------------------------------------- #
# Classification labels -- pluggable interface
# --------------------------------------------------------------------------- #
class LabelGenerator(ABC):
    """Interface for a pluggable classification labeling rule.

    Implementations receive the full historical DataFrame plus a decision
    index and horizon, and return one label string from :attr:`classes`.
    Subclass this to plug in a custom labeling strategy (e.g. triple-barrier,
    ATR-scaled thresholds, fixed take-profit/stop-loss) without touching
    :class:`~ml_pipeline.dataset_builder.DatasetBuilder`.
    """

    classes: ClassVar[Tuple[str, ...]]

    @abstractmethod
    def label(self, df: pd.DataFrame, index: int, horizon: int) -> str:
        """Return one label from ``self.classes`` for the candle at ``index``."""
        raise NotImplementedError


class ThresholdLabelGenerator(LabelGenerator):
    """Default labeling rule: BUY / SELL / NO_TRADE from a forward-return threshold.

    Parameters
    ----------
    buy_threshold:
        Forward return strictly above this fraction => ``"BUY"``.
    sell_threshold:
        Forward return strictly below this fraction => ``"SELL"``.
    """

    classes: ClassVar[Tuple[str, ...]] = ("SELL", "NO_TRADE", "BUY")

    def __init__(self, buy_threshold: float = 0.0005, sell_threshold: float = -0.0005) -> None:
        if buy_threshold <= 0:
            raise ValueError("buy_threshold must be > 0")
        if sell_threshold >= 0:
            raise ValueError("sell_threshold must be < 0")
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def label(self, df: pd.DataFrame, index: int, horizon: int) -> str:
        ret = _next_return(df, index, horizon, pip_size=0.0001)
        if ret > self.buy_threshold:
            return "BUY"
        if ret < self.sell_threshold:
            return "SELL"
        return "NO_TRADE"


CLASSIFICATION_REGISTRY: Dict[str, Type[LabelGenerator]] = {
    "threshold": ThresholdLabelGenerator,
}


def build_classification_label_generator(name: str, **kwargs) -> LabelGenerator:
    """Resolve a registered classification label generator by name.

    Register a custom rule with ``CLASSIFICATION_REGISTRY["my_rule"] = MyGenerator``,
    or bypass the registry entirely by constructing a :class:`LabelGenerator`
    and passing it directly to ``DatasetBuilder(label_generator=...)``.
    """
    if name not in CLASSIFICATION_REGISTRY:
        raise ValueError(
            f"Unknown classification_label {name!r}. "
            f"Registered: {sorted(CLASSIFICATION_REGISTRY)}"
        )
    return CLASSIFICATION_REGISTRY[name](**kwargs)
