"""AnthropicLLMClient tests — mock the SDK to avoid network calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock, ToolUseBlock
from pydantic import SecretStr

from annrag.llm.anthropic_client import AnthropicLLMClient
from annrag.llm.base import LLMError


def _make_client_with_response(monkeypatch, content_blocks, *, stop_reason="tool_use"):
    """Patch the underlying anthropic SDK with a stub that returns `content_blocks`."""
    captured: dict = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=content_blocks, stop_reason=stop_reason)

    class FakeAnthropic:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    monkeypatch.setattr("annrag.llm.anthropic_client.Anthropic", FakeAnthropic)
    client = AnthropicLLMClient(api_key=SecretStr("sk-test"), model="claude-test-1")
    return client, captured


def test_returns_tool_input_dict(monkeypatch):
    block = ToolUseBlock(id="tu1", name="my_tool", type="tool_use", input={"k": "v"})
    client, captured = _make_client_with_response(monkeypatch, [block])
    out = client.call_tool(
        system="sys",
        user="usr",
        tool_name="my_tool",
        tool_description="desc",
        input_schema={"type": "object"},
        max_tokens=42,
    )
    assert out == {"k": "v"}
    assert captured["model"] == "claude-test-1"
    assert captured["max_tokens"] == 42
    assert captured["tools"][0]["name"] == "my_tool"
    assert captured["tool_choice"] == {"type": "tool", "name": "my_tool"}
    assert captured["api_key"] == "sk-test"


def test_raises_when_model_doesnt_call_tool(monkeypatch):
    text_block = TextBlock(type="text", text="hi", citations=None)
    client, _ = _make_client_with_response(monkeypatch, [text_block], stop_reason="end_turn")
    with pytest.raises(LLMError, match="did not invoke tool"):
        client.call_tool(
            system="s",
            user="u",
            tool_name="t",
            tool_description="d",
            input_schema={},
        )


def test_raises_when_tool_input_not_dict(monkeypatch):
    # Construct a ToolUseBlock whose input passes the SDK validator but our
    # stricter dict check rejects (the SDK accepts anything JSON-serializable).
    block = ToolUseBlock.model_construct(id="tu1", name="t", type="tool_use", input="not-a-dict")
    client, _ = _make_client_with_response(monkeypatch, [block])
    with pytest.raises(LLMError, match="not a dict"):
        client.call_tool(
            system="s",
            user="u",
            tool_name="t",
            tool_description="d",
            input_schema={},
        )


def test_sdk_failure_wrapped(monkeypatch):
    class FakeMessages:
        def create(self, **_kwargs):
            raise RuntimeError("network down")

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setattr("annrag.llm.anthropic_client.Anthropic", FakeAnthropic)
    client = AnthropicLLMClient(api_key=SecretStr("k"), model="m")
    with pytest.raises(LLMError, match="anthropic call failed"):
        client.call_tool(
            system="s",
            user="u",
            tool_name="t",
            tool_description="d",
            input_schema={},
        )
