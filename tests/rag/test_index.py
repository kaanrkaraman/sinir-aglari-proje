"""Tests for M4 vector index — FAISS mocked for CI compatibility."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

from annrag.rag.models import Chunk


DIM = 8


def make_chunk(idx: int, page: str = "Paris") -> Chunk:
    return Chunk(
        chunk_id=f"{page}::{idx}",
        source_page=page,
        text=f"Sample text for chunk {idx}.",
        start_char=idx * 100,
        end_char=idx * 100 + 50,
        token_count=10,
        strategy="fixed_512",
    )


def make_vector(dim: int = DIM, seed: int = 0) -> list[float]:
    return [math.sin(seed + i) for i in range(dim)]


# ── OllamaEmbedder tests (mocked — no Ollama needed) ─────────────────────────

def test_embedder_embed_calls_api():
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": make_vector(768)}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from annrag.rag.embedder import OllamaEmbedder
        embedder = OllamaEmbedder(model="nomic-embed-text")
        vec = embedder.embed("Hello world")

        assert len(vec) == 768
        mock_post.assert_called_once()


def test_embedder_returns_floats():
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": make_vector(768)}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from annrag.rag.embedder import OllamaEmbedder
        embedder = OllamaEmbedder()
        vec = embedder.embed("test")

        assert all(isinstance(v, float) for v in vec)


def test_embedder_batch_length():
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": make_vector(768)}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from annrag.rag.embedder import OllamaEmbedder
        embedder = OllamaEmbedder()
        texts = ["text one", "text two", "text three"]
        vecs = embedder.embed_batch(texts)

        assert len(vecs) == 3
        assert mock_post.call_count == 3


def test_embedder_model_name():
    from annrag.rag.embedder import OllamaEmbedder
    embedder = OllamaEmbedder(model="nomic-embed-text")
    assert embedder.model == "nomic-embed-text"


# ── Chunk model tests ─────────────────────────────────────────────────────────

def test_chunk_immutable():
    import pytest
    chunk = make_chunk(0)
    with pytest.raises(Exception):
        chunk.text = "new text"  # type: ignore


def test_chunk_id_format():
    chunk = make_chunk(3, page="Tokyo")
    assert chunk.chunk_id == "Tokyo::3"


def test_chunk_source_page():
    chunk = make_chunk(0, page="Amsterdam")
    assert chunk.source_page == "Amsterdam"
