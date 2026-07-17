"""Configuration for the ML dataset-building pipeline.

All tunable parameters live in :class:`DatasetConfig`, mirroring the
Market Structure Engine's own ``EngineConfig`` convention: every threshold
is explicit and overridable, nothing is read from global state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from market_structure import EngineConfig

ScalerMethod = Literal["standard", "minmax", "robust", "none"]
SelectorMethod = Literal["variance", "correlation", "mutual_info", "rfe", "kbest"]
TargetType = Literal["regression", "classification"]

DEFAULT_REGRESSION_TARGETS: tuple[str, ...] = ("next_close", "next_return")

REGRESSION_TARGET_NAMES: tuple[str, ...] = (
    "next_close",
    "next_open",
    "next_high",
    "next_low",
    "next_return",
    "next_log_return",
    "expected_pip_movement",
    "expected_percentage_change",
    "future_atr",
    "future_volatility",
)


@dataclass(slots=True)
class DatasetConfig:
    """Tunable parameters for :class:`ml_pipeline.dataset_builder.DatasetBuilder`.

    Attributes
    ----------
    symbol, timeframe:
        Free-form labels stamped into every sample's metadata (e.g.
        ``"EUR_USD"``, ``"M5"``). Not used for any computation.
    window_size:
        Number of trailing candles fed to the Market Structure Engine for
        each sample (``EngineConfig`` requires >= 3; this pipeline defaults
        higher so every warm-up-gated feature in the engine's ``_valid``
        flags has a realistic chance of being warmed up).
    horizon:
        Bars ahead of the decision candle used to compute labels. Must be
        >= 1 -- this is the only place future data is intentionally read
        (for the *label*, never the *features*).
    stride:
        Step between consecutive decision candles. 1 = one sample per
        historical candle (as specified); > 1 skips candles to bound
        runtime on very long histories.
    engine_config:
        Passed through verbatim to every ``MarketStructureEngine`` instance
        this builder creates. The engine itself is never modified.
    regression_targets:
        Names from :data:`REGRESSION_TARGET_NAMES` to compute for every
        sample. Each becomes one column of ``Dataset.y_reg``.
    classification_label:
        Name of the classification labeling rule. ``"threshold"`` uses the
        built-in :class:`~ml_pipeline.label_generator.ThresholdLabelGenerator`;
        pass a custom :class:`~ml_pipeline.label_generator.LabelGenerator`
        instance directly to ``DatasetBuilder`` to override.
    classification_kwargs:
        Keyword arguments forwarded to the classification label generator's
        constructor (e.g. ``{"buy_threshold": 0.0008}``).
    impute_invalid:
        When True, :class:`~ml_pipeline.feature_pipeline.FeaturePipeline`
        replaces a feature's value with the training-set median wherever
        its paired ``_valid`` flag is 0.0, instead of leaving the engine's
        raw placeholder (usually 0.0). The ``_valid`` flag itself is never
        removed, so a model can still learn to weight it.
    one_hot_categorical:
        When True, one-hot encode the engine's signed categorical features
        (e.g. ``trend_direction`` in {-1,0,1}) instead of feeding the raw
        code as a single ordinal column.
    cyclical_time_encoding:
        When True, replace ``session_hour``/``session_minute``/
        ``session_day_of_week``/``session_month`` with sin/cos pairs.
    scaler:
        One of ``"standard"``, ``"minmax"``, ``"robust"``, ``"none"``.
    feature_selector:
        Optional name from :data:`SelectorMethod`; ``None`` disables
        selection (all engine features are kept).
    feature_selector_kwargs:
        Keyword arguments forwarded to
        :class:`~ml_pipeline.feature_selector.FeatureSelector`.
    random_state:
        Seed forwarded to any randomized component (currently only RFE's
        internal estimator, if used).
    """

    symbol: str = "UNKNOWN"
    timeframe: str = "UNKNOWN"
    window_size: int = 500
    horizon: int = 1
    stride: int = 1
    engine_config: EngineConfig = field(default_factory=EngineConfig)
    regression_targets: List[str] = field(
        default_factory=lambda: list(DEFAULT_REGRESSION_TARGETS)
    )
    classification_label: str = "threshold"
    classification_kwargs: Dict[str, Any] = field(default_factory=dict)
    impute_invalid: bool = True
    one_hot_categorical: bool = False
    cyclical_time_encoding: bool = False
    scaler: ScalerMethod = "standard"
    feature_selector: Optional[SelectorMethod] = None
    feature_selector_kwargs: Dict[str, Any] = field(default_factory=dict)
    pip_size: float = 0.0001
    """Price units per pip, for the ``expected_pip_movement`` regression target
    (0.0001 for most FX pairs; use 0.01 for JPY-quoted pairs)."""
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.window_size < 3:
            raise ValueError("window_size must be >= 3 (engine requires >= 3 candles).")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1 -- a horizon of 0 has no future to predict.")
        if self.stride < 1:
            raise ValueError("stride must be >= 1.")
        unknown = set(self.regression_targets) - set(REGRESSION_TARGET_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown regression target(s): {sorted(unknown)}. "
                f"Valid names: {REGRESSION_TARGET_NAMES}"
            )
