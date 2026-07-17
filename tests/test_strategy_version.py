"""Tests for strategy.strategy_version."""
from __future__ import annotations

import pytest

from strategy.strategy_version import StrategyVersion, bump_version


def test_new_populates_all_fields() -> None:
    v = StrategyVersion.new("ict")
    assert v.strategy_name == "ict"
    assert v.strategy_version == "1.0.0"
    assert v.rule_version
    assert v.configuration_version == "1.0.0"
    assert v.timestamp


def test_rejects_non_semver() -> None:
    with pytest.raises(ValueError):
        StrategyVersion.new("ict", strategy_version="not-a-version")


def test_dict_round_trip() -> None:
    v = StrategyVersion.new("ict")
    restored = StrategyVersion.from_dict(v.to_dict())
    assert restored == v


def test_bump_version_patch_minor_major() -> None:
    assert bump_version("1.2.3", "patch") == "1.2.4"
    assert bump_version("1.2.3", "minor") == "1.3.0"
    assert bump_version("1.2.3", "major") == "2.0.0"


def test_bump_version_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        bump_version("1.2", "patch")
    with pytest.raises(ValueError):
        bump_version("1.2.3", "bogus")
