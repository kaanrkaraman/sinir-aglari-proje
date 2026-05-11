"""LLM client abstraction.

Components consume the `LLMClient` Protocol so we can swap providers
(Anthropic / Ollama / OpenAI / ...) without touching the call sites that
produce synthetic Q&A in M2 or generate RAG answers in M6.
"""

from annrag.llm.base import LLMClient, LLMError
from annrag.llm.factory import ConfigurationError, build_llm_client

__all__ = ["ConfigurationError", "LLMClient", "LLMError", "build_llm_client"]
