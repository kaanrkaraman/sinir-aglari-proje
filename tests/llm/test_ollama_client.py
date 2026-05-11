"""OllamaLLMClient tests — patch the SDK so no daemon is needed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from annrag.llm.base import LLMError
from annrag.llm.ollama_client import OllamaLLMClient, list_available_models


def _patch_ollama_client(monkeypatch, *, response, capture: dict | None = None):
    """Replace `ollama.Client` with a stub that returns `response` from chat()."""
    captured = capture if capture is not None else {}

    class FakeClient:
        def __init__(self, host, timeout=None, **_kwargs):
            captured["host"] = host
            captured["timeout"] = timeout

        def chat(self, **kwargs):
            captured.update(kwargs)
            if isinstance(response, Exception):
                raise response
            return response

        def list(self):
            return SimpleNamespace(
                models=[
                    SimpleNamespace(model="llama3:8b"),
                    SimpleNamespace(model="mistral:7b"),
                    SimpleNamespace(model=""),  # filtered out
                ]
            )

    monkeypatch.setattr("annrag.llm.ollama_client.ollama.Client", FakeClient)
    return captured


def _msg(content):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def test_returns_parsed_dict(monkeypatch):
    captured = _patch_ollama_client(monkeypatch, response=_msg('{"k": "v", "n": 3}'))
    client = OllamaLLMClient(base_url="http://x:1", model="llama3:8b")
    out = client.call_tool(
        system="sys",
        user="usr",
        tool_name="t",
        tool_description="d",
        input_schema={"type": "object"},
        max_tokens=42,
    )
    assert out == {"k": "v", "n": 3}
    # The schema is forwarded to Ollama as `format=`.
    assert captured["format"] == {"type": "object"}
    assert captured["model"] == "llama3:8b"
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["content"] == "sys"
    assert captured["messages"][1]["role"] == "user"
    assert captured["options"]["num_predict"] == 42
    assert captured["host"] == "http://x:1"


def test_empty_content_raises(monkeypatch):
    _patch_ollama_client(monkeypatch, response=_msg(""))
    client = OllamaLLMClient(base_url="http://x:1", model="llama3:8b")
    with pytest.raises(LLMError, match="empty content"):
        client.call_tool(system="s", user="u", tool_name="t", tool_description="d", input_schema={})


def test_malformed_json_raises(monkeypatch):
    _patch_ollama_client(monkeypatch, response=_msg("not-json"))
    client = OllamaLLMClient(base_url="http://x:1", model="llama3:8b")
    with pytest.raises(LLMError, match="malformed JSON"):
        client.call_tool(system="s", user="u", tool_name="t", tool_description="d", input_schema={})


def test_non_dict_json_raises(monkeypatch):
    _patch_ollama_client(monkeypatch, response=_msg("[1, 2, 3]"))
    client = OllamaLLMClient(base_url="http://x:1", model="llama3:8b")
    with pytest.raises(LLMError, match="non-dict"):
        client.call_tool(system="s", user="u", tool_name="t", tool_description="d", input_schema={})


def test_sdk_exception_wrapped(monkeypatch):
    _patch_ollama_client(monkeypatch, response=RuntimeError("connection refused"))
    client = OllamaLLMClient(base_url="http://x:1", model="llama3:8b")
    with pytest.raises(LLMError, match="ollama call failed"):
        client.call_tool(system="s", user="u", tool_name="t", tool_description="d", input_schema={})


def test_list_available_models_filters_empty(monkeypatch):
    _patch_ollama_client(monkeypatch, response=_msg("{}"))
    tags = list_available_models("http://x:1")
    assert tags == ["llama3:8b", "mistral:7b"]
