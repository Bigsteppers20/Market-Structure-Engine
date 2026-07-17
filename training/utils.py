"""Shared, dependency-free utilities for the training infrastructure.

Mirrors the convention already used by ``market_structure/utils.py`` and
``ml_pipeline``: small helpers reused across modules, no module-level state.
"""
from __future__ import annotations

import hashlib
import random
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def utc_timestamp() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str = "exp") -> str:
    """A short, collision-resistant identifier: ``{prefix}_{12 hex chars}``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def get_git_commit(repo_dir: Optional[Path] = None) -> Optional[str]:
    """Best-effort current git commit hash, or ``None`` if unavailable.

    Never raises: returns ``None`` when the directory isn't a git repository,
    git isn't installed, or the lookup times out -- a training run must be
    able to proceed and be logged even outside version control.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def hash_strings(values: Sequence[str]) -> str:
    """Stable short hash of an ordered sequence of strings (e.g. feature names)."""
    joined = "\x1f".join(values)  # unit separator -- avoids delimiter collisions
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def set_random_seed(seed: int) -> None:
    """Seed every RNG this infrastructure touches, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


class Timer:
    """Context manager measuring wall-clock elapsed seconds.

    Usage::

        with Timer() as t:
            do_work()
        print(t.elapsed_seconds)
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed_seconds: float = 0.0
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start


def percentile(values: List[float], pct: float) -> float:
    """Linear-interpolated percentile, avoiding a numpy import at call sites."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing; return it for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path
