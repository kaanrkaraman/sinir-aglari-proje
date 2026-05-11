"""Anthropic implementation of the `LLMClient` protocol.

Uses the Messages API with a forced single-tool call to produce strictly
structured output. The schema is defined by the caller (see
`groundtruth.generate.QA_TOOL_SCHEMA`); we just pass it through.
"""

from __future__ import annotations

import logging
from typing import Any

from anthropic import Anthropic
from anthropic.types import ToolUseBlock
from pydantic import SecretStr

from annrag.llm.base import LLMError

logger = logging.getLogger(__name__)


class AnthropicLLMClient:
    """Thin wrapper over `anthropic.Anthropic` exposing `LLMClient.call_tool`."""

    def __init__(self, api_key: SecretStr, model: str) -> None:
        self._client = Anthropic(api_key=api_key.get_secret_value())
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def call_tool(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                tools=[
                    {
                        "name": tool_name,
                        "description": tool_description,
                        "input_schema": input_schema,
                    }
                ],
                tool_choice={"type": "tool", "name": tool_name},
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:
            raise LLMError(f"anthropic call failed: {e}") from e

        for block in response.content:
            if isinstance(block, ToolUseBlock) and block.name == tool_name:
                payload = block.input
                if not isinstance(payload, dict):
                    raise LLMError(f"tool_use input was not a dict: type={type(payload).__name__}")
                return dict(payload)
        raise LLMError(
            f"model {self._model!r} did not invoke tool {tool_name!r}; "
            f"stop_reason={response.stop_reason!r}"
        )
