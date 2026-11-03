"""Shared fixtures.

Every test runs against a temporary profile directory so nothing touches the
developer's real ``~/.aura``.
"""

from __future__ import annotations

import pytest

from aura.config import EngineConfig
from aura.pipeline.engine import AuraEngine
from helpers import HOP, SR  # noqa: F401  (re-exported for convenience)


@pytest.fixture
def config(tmp_path) -> EngineConfig:
    return EngineConfig(
        sample_rate=SR,
        block_size=HOP,
        profile_dir=str(tmp_path / "profile"),
        publish_rate_hz=1000.0,
    )


@pytest.fixture
def engine(config) -> AuraEngine:
    eng = AuraEngine(config)
    yield eng
    eng.stop()
