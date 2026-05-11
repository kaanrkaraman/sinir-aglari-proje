"""Provider-agnostic LLM client construction."""

from __future__ import annotations

from annrag.config import LLMProvider, Settings
from annrag.llm.anthropic_client import AnthropicLLMClient
from annrag.llm.base import LLMClient
from annrag.llm.ollama_client import OllamaLLMClient


class ConfigurationError(RuntimeError):
    """Raised when the provider is selected but its required config is missing."""


def build_llm_client(
    settings: Settings,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> tuple[LLMClient, str]:
    """Build a client for the (possibly overridden) provider/model.

    Returns `(client, model_id)` so the caller (e.g. the Q&A generator) can
    record which model produced each candidate without having to crack open
    the client implementation.
    """
    chosen = provider or settings.llm_provider

    if chosen == "anthropic":
        if settings.anthropic_api_key is None:
            raise ConfigurationError("llm_provider=anthropic requires ANTHROPIC_API_KEY in .env")
        model_id = model or settings.anthropic_model
        return AnthropicLLMClient(
            api_key=settings.anthropic_api_key,
            model=model_id,
        ), model_id

    if chosen == "ollama":
        model_id = model or settings.ollama_model
        return OllamaLLMClient(
            base_url=settings.ollama_base_url,
            model=model_id,
            request_timeout_s=settings.ollama_request_timeout_s,
        ), model_id

    raise ConfigurationError(f"unknown llm provider: {chosen!r}")
