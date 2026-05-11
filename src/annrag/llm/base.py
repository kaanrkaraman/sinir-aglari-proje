"""LLM client Protocol — provider-agnostic surface used by the rest of annrag."""

from __future__ import annotations

from typing import Any, Protocol


class LLMError(RuntimeError):
    """Raised when an LLM call fails or returns an unusable response."""


class LLMClient(Protocol):
    """Minimum surface every LLM provider must expose for annrag.

    `call_tool` is the only method needed for M2 (forced structured output via
    a single tool call). M6 will extend this with `complete` for free-form
    answer generation.
    """

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
        """Force the model to call `tool_name`; return the tool's `input` dict.

        Implementations must raise `LLMError` if the model fails to invoke the
        tool, returns malformed args, or hits an unrecoverable API error.
        """
        ...
