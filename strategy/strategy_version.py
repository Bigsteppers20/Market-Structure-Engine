"""Strategy versioning.

Every strategy carries four independent version numbers (name, strategy
version, rule version, configuration version) plus a timestamp -- mirroring
the same discipline already established in ``training.versioning`` for
trained models, applied here to strategy definitions instead.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal

from training.utils import utc_timestamp

#: Version of the built-in rule library in ``strategy/rule_base.py``. Bump
#: this whenever a built-in rule's pass/fail logic changes in a way that
#: could change historical evaluation results.
RULE_LIBRARY_VERSION = "1.0.0"

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True, slots=True)
class StrategyVersion:
    """Full version identity of one strategy definition."""

    strategy_name: str
    strategy_version: str
    rule_version: str
    configuration_version: str
    timestamp: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("strategy_version", self.strategy_version),
            ("rule_version", self.rule_version),
            ("configuration_version", self.configuration_version),
        ):
            if not _SEMVER_RE.match(value):
                raise ValueError(
                    f"{field_name}={value!r} is not a semantic version (expected 'X.Y.Z')."
                )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyVersion":
        return cls(**d)

    @classmethod
    def new(
        cls, strategy_name: str, strategy_version: str = "1.0.0",
        rule_version: str = RULE_LIBRARY_VERSION, configuration_version: str = "1.0.0",
    ) -> "StrategyVersion":
        return cls(
            strategy_name=strategy_name, strategy_version=strategy_version,
            rule_version=rule_version, configuration_version=configuration_version,
            timestamp=utc_timestamp(),
        )


def bump_version(version: str, part: Literal["major", "minor", "patch"] = "patch") -> str:
    """Increment one component of a semantic version string, e.g. ``"1.2.3"``."""
    if not _SEMVER_RE.match(version):
        raise ValueError(f"{version!r} is not a semantic version (expected 'X.Y.Z').")
    major, minor, patch = (int(p) for p in version.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"part must be 'major', 'minor', or 'patch', got {part!r}.")
