"""Classification label generation.

Implements ``ml_pipeline.label_generator.LabelGenerator`` -- the pluggable
labeling interface the Dataset Builder was explicitly designed around --
rather than inventing a parallel labeling mechanism. This is the *intended*
extension point (see ``ml_pipeline``'s own docs: "subclass this to plug in
a custom labeling strategy"), not a duplication of it.

The default 3-class rule (SELL / NO_TRADE / BUY) considers, all
configurably: minimum pip movement, minimum expected return, a
risk-reward ratio (favorable vs. adverse excursion within the horizon), and
an optional cap on adverse excursion. The class *set* itself is a
constructor parameter, not hardcoded -- extending to
STRONG_BUY/WEAK_BUY/EXIT_LONG/... means constructing this (or a thin
subclass) with a different ``classes`` tuple and threshold scheme, never a
change to the Dataset Builder or this class's public shape.
"""
from __future__ import annotations

from typing import ClassVar, Tuple

import pandas as pd
from ml_pipeline.label_generator import LabelGenerator

DEFAULT_CLASSES: Tuple[str, ...] = ("SELL", "NO_TRADE", "BUY")


def _forward_return(df: pd.DataFrame, index: int, horizon: int) -> float:
    c0 = float(df["close"].iloc[index])
    c1 = float(df["close"].iloc[index + horizon])
    return (c1 - c0) / c0 if c0 else 0.0


def _forward_pip_movement(df: pd.DataFrame, index: int, horizon: int, pip_size: float) -> float:
    c0 = float(df["close"].iloc[index])
    c1 = float(df["close"].iloc[index + horizon])
    return (c1 - c0) / pip_size if pip_size else 0.0


def _favorable_and_adverse_excursion(df: pd.DataFrame, index: int, horizon: int) -> tuple[float, float]:
    """Long-bias convention (see linear_regression's identical formula,
    independently defined here to keep this engine self-contained --
    see module docstring / LOGISTIC_REGRESSION_ENGINE_REPORT.md):
    MFE = best-case favorable move, MAE = worst-case adverse move, both
    non-negative price-unit magnitudes."""
    end = min(index + 1 + horizon, len(df))
    window = df.iloc[index + 1: end]
    entry = float(df["close"].iloc[index])
    if window.empty:
        return 0.0, 0.0
    mfe = float(window["high"].max() - entry)
    mae = float(entry - window["low"].min())
    return max(mfe, 0.0), max(mae, 0.0)


class ConfigurableClassificationLabelGenerator(LabelGenerator):
    """Default labeling rule: pip movement + return + risk-reward + optional
    adverse-excursion cap, evaluated symmetrically for a long (BUY) and
    short (SELL) hypothesis.

    Parameters
    ----------
    classes:
        Ordered class names. Only the exact default ``(SELL, NO_TRADE, BUY)``
        (in any order) is scored by the built-in rule below; pass a
        different set together with a subclass overriding :meth:`label` to
        extend (see ``tests/test_lgr_label_manager.py`` for a worked
        5-class example).
    min_pip_movement:
        Minimum |pip movement| over the horizon to consider a direction.
    min_expected_return:
        Minimum |fractional return| over the horizon to consider a direction.
    risk_reward_threshold:
        Minimum favorable/adverse excursion ratio required.
    max_adverse_excursion_pips:
        Optional hard cap: reject a direction if its adverse excursion
        (in pips) exceeds this.
    pip_size:
        Price units per pip.
    """

    def __init__(
        self,
        classes: Tuple[str, ...] = DEFAULT_CLASSES,
        min_pip_movement: float = 5.0,
        min_expected_return: float = 0.0,
        risk_reward_threshold: float = 1.5,
        max_adverse_excursion_pips: float | None = None,
        pip_size: float = 0.0001,
    ) -> None:
        if len(set(classes)) != len(classes) or len(classes) < 2:
            raise ValueError(f"classes must be a set of >= 2 unique names, got {classes!r}.")
        self.classes: ClassVar[Tuple[str, ...]] = tuple(classes)  # instance override, see strategy/rule_base precedent
        self.min_pip_movement = min_pip_movement
        self.min_expected_return = min_expected_return
        self.risk_reward_threshold = risk_reward_threshold
        self.max_adverse_excursion_pips = max_adverse_excursion_pips
        self.pip_size = pip_size
        self._buy = self._find_class(classes, "BUY")
        self._sell = self._find_class(classes, "SELL")
        self._no_trade = self._find_class(classes, "NO_TRADE")

    @staticmethod
    def _find_class(classes: Tuple[str, ...], name: str) -> str:
        if name not in classes:
            raise ValueError(
                f"ConfigurableClassificationLabelGenerator's default rule requires a {name!r} "
                f"class; got {classes!r}. Subclass and override label() for a custom class set."
            )
        return name

    def label(self, df: pd.DataFrame, index: int, horizon: int) -> str:
        ret = _forward_return(df, index, horizon)
        pip_move = _forward_pip_movement(df, index, horizon, self.pip_size)
        mfe, mae = _favorable_and_adverse_excursion(df, index, horizon)

        long_rr = (mfe / mae) if mae > 1e-12 else (float("inf") if mfe > 0 else 0.0)
        # Short hypothesis: "favorable" is the downward move (mae, in the
        # long-bias formula) and "adverse" is the upward move (mfe).
        short_rr = (mae / mfe) if mfe > 1e-12 else (float("inf") if mae > 0 else 0.0)

        long_ok = (
            pip_move >= self.min_pip_movement
            and ret >= self.min_expected_return
            and long_rr >= self.risk_reward_threshold
        )
        short_ok = (
            -pip_move >= self.min_pip_movement
            and -ret >= self.min_expected_return
            and short_rr >= self.risk_reward_threshold
        )
        if self.max_adverse_excursion_pips is not None:
            mae_pips = mae / self.pip_size if self.pip_size else 0.0
            mfe_pips = mfe / self.pip_size if self.pip_size else 0.0
            long_ok = long_ok and mae_pips <= self.max_adverse_excursion_pips
            short_ok = short_ok and mfe_pips <= self.max_adverse_excursion_pips

        if long_ok and not short_ok:
            return self._buy
        if short_ok and not long_ok:
            return self._sell
        return self._no_trade
