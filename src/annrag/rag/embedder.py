"""Embedding module — M4.

Calls Ollama's nomic-embed-text model to convert chunk text into vectors.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"


@runtime_checkable
class Embedder(Protocol):
    """Any callable that turns text into a float vector."""

    @property
    def model(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """Embeds text using Ollama's local embedding model.

    Args:
        model: Ollama model name (default: nomic-embed-text).
        base_url: Ollama server URL.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._dim: int | None = None
        self._client = httpx.Client(timeout=60.0)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Embed a dummy text to discover dimension
            vec = self.embed("hello")
            self._dim = len(vec)
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        # nomic-embed-text has a token limit — truncate to be safe
        text = text[:2000]
        resp = self._client.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts one by one."""
        vectors = []
        for i, text in enumerate(texts):
            vec = self.embed(text)
            vectors.append(vec)
            if (i + 1) % 50 == 0:
                logger.info("Embedded %d / %d", i + 1, len(texts))
        return vectors
