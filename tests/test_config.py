"""Smoke tests for Settings — defaults, env override, validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from annrag.config import Settings


def test_defaults_load(tmp_path):
    s = Settings()
    assert s.log_level == "INFO"
    assert s.random_seed == 42
    # Paths are absolute after the validator resolves them.
    assert s.data_dir.is_absolute()
    assert s.artifacts_dir.is_absolute()


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ANNRAG_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ANNRAG_RANDOM_SEED", "123")
    s = Settings()
    assert s.log_level == "DEBUG"
    assert s.random_seed == 123


def test_invalid_log_level_rejected(monkeypatch):
    monkeypatch.setenv("ANNRAG_LOG_LEVEL", "TRACE")
    with pytest.raises(ValidationError):
        Settings()


def test_negative_seed_rejected(monkeypatch):
    monkeypatch.setenv("ANNRAG_RANDOM_SEED", "-1")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_is_frozen():
    s = Settings()
    with pytest.raises(ValidationError):
        s.log_level = "DEBUG"  # type: ignore[misc]
