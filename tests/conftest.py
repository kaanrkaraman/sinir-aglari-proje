"""Shared pytest fixtures.

Isolates Settings from the host environment so tests are deterministic
regardless of what `ANNRAG_*` vars or `.env` files exist on the dev machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ANNRAG_ENV_VARS = (
    "ANNRAG_DATA_DIR",
    "ANNRAG_ARTIFACTS_DIR",
    "ANNRAG_LOG_LEVEL",
    "ANNRAG_RANDOM_SEED",
)


@pytest.fixture(autouse=True)
def _isolate_settings_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Strip any host ANNRAG_* env vars and run from an empty cwd so a stray
    # `.env` in the repo doesn't bleed into tests.
    for key in _ANNRAG_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
