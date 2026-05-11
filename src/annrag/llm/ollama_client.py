"""Ollama implementation of the `LLMClient` protocol.

Ollama exposes structured output as a first-class feature: passing a JSON
schema as `format=...` constrains the response to conform. We map the
provider-agnostic `call_tool` semantic ("produce data matching this schema")
onto that mechanism — no actual tool dispatch, just schema-constrained JSON.
This is more reliable than tool-use on smaller local models, which often
struggle with multi-turn tool calling.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import ollama

from annrag.llm.base import LLMError

logger = logging.getLogger(__name__)


class OllamaLLMClient:
    """LLM client that talks to a local (or remote) Ollama daemon."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        request_timeout_s: float = 300.0,
        temperature: float = 0.7,
    ) -> None:
        self._client = ollama.Client(host=base_url, timeout=request_timeout_s)
        self._model = model
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    def call_tool(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,  # kept for Protocol parity; Ollama uses format=schema instead.
        tool_description: str,  # same
        input_schema: dict[str, Any],
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        del tool_name, tool_description  # silence ruff/ty about unused params
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format=input_schema,
                options={
                    "temperature": self._temperature,
                    "num_predict": max_tokens,
                },
            )
        except Exception as e:  # SDK can raise httpx, ollama.ResponseError, etc.
            raise LLMError(f"ollama call failed for model={self._model!r}: {e}") from e

        content = response.message.content
        if not content:
            raise LLMError(f"ollama returned empty content (model={self._model!r})")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMError(
                f"ollama returned malformed JSON (model={self._model!r}): {e}; "
                f"content[:200]={content[:200]!r}"
            ) from e
        if not isinstance(parsed, dict):
            raise LLMError(
                f"ollama returned non-dict JSON (model={self._model!r}): "
                f"type={type(parsed).__name__}"
            )
        return parsed


def list_available_models(base_url: str) -> list[str]:
    """Return the tags of every model installed in the local Ollama daemon."""
    client = ollama.Client(host=base_url)
    payload = client.list()
    # SDK returns ListResponse with `.models: list[Model]`; each has `.model`.
    return [m.model for m in payload.models if m.model]
