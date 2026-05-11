"""LLM factory tests."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from annrag.config import Settings
from annrag.llm import build_llm_client
from annrag.llm.anthropic_client import AnthropicLLMClient
from annrag.llm.factory import ConfigurationError
from annrag.llm.ollama_client import OllamaLLMClient


def test_builds_ollama_by_default(monkeypatch):
    # Patch the SDK constructors so we don't actually open a connection.
    monkeypatch.setattr("annrag.llm.ollama_client.ollama.Client", lambda **_kw: object())
    settings = Settings(llm_provider="ollama")
    client, model = build_llm_client(settings)
    assert isinstance(client, OllamaLLMClient)
    assert model == settings.ollama_model


def test_builds_anthropic_when_keyed(monkeypatch):
    monkeypatch.setattr("annrag.llm.anthropic_client.Anthropic", lambda api_key: object())
    settings = Settings(
        llm_provider="anthropic",
        anthropic_api_key=SecretStr("sk-test"),
    )
    client, model = build_llm_client(settings)
    assert isinstance(client, AnthropicLLMClient)
    assert model == settings.anthropic_model


def test_anthropic_missing_key_raises():
    settings = Settings(llm_provider="anthropic")
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        build_llm_client(settings)


def test_provider_override(monkeypatch):
    monkeypatch.setattr("annrag.llm.ollama_client.ollama.Client", lambda **_kw: object())
    settings = Settings(llm_provider="anthropic")  # default would fail
    client, model = build_llm_client(settings, provider="ollama", model="mistral:7b")
    assert isinstance(client, OllamaLLMClient)
    assert model == "mistral:7b"
