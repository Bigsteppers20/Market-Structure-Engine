"""Tests for decision_engine.config and decision_engine.version."""
from __future__ import annotations

import pytest

from decision_engine.config import DecisionEngineConfig
from decision_engine.exceptions import InvalidConfigError
from decision_engine.version import DECISION_ENGINE_VERSION


def test_defaults() -> None:
    cfg = DecisionEngineConfig()
    assert cfg.pip_size == 0.0001
    assert cfg.stop_atr_multiple == 1.5
    assert cfg.take_profit_r_multiples == (1.0, 2.0, 3.0)


def test_rejects_non_positive_pip_size() -> None:
    with pytest.raises(InvalidConfigError):
        DecisionEngineConfig(pip_size=0.0)


def test_rejects_non_positive_stop_atr_multiple() -> None:
    with pytest.raises(InvalidConfigError):
        DecisionEngineConfig(stop_atr_multiple=-1.0)


def test_rejects_wrong_number_of_take_profit_multiples() -> None:
    with pytest.raises(InvalidConfigError):
        DecisionEngineConfig(take_profit_r_multiples=(1.0, 2.0))


def test_rejects_non_increasing_take_profit_multiples() -> None:
    with pytest.raises(InvalidConfigError):
        DecisionEngineConfig(take_profit_r_multiples=(2.0, 1.0, 3.0))


def test_rejects_out_of_range_downgrade_threshold() -> None:
    with pytest.raises(InvalidConfigError):
        DecisionEngineConfig(downgrade_opposition_threshold=1.5)


def test_to_dict_from_dict_round_trip() -> None:
    cfg = DecisionEngineConfig(stop_atr_multiple=2.0, take_profit_r_multiples=(0.5, 1.5, 2.5))
    restored = DecisionEngineConfig.from_dict(cfg.to_dict())
    assert restored.stop_atr_multiple == 2.0
    assert restored.take_profit_r_multiples == (0.5, 1.5, 2.5)


def test_decision_engine_version_is_a_string() -> None:
    assert isinstance(DECISION_ENGINE_VERSION, str)
    assert DECISION_ENGINE_VERSION
