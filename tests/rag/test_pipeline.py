"""Tests for M6 RAG pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from annrag.rag.pipeline import RAGResponse, build_context, generate_answer


def make_chunk(page: str = "Paris", text: str = "Paris is beautiful."):
    from annrag.rag.models import Chunk
    return Chunk(
        chunk_id=f"{page}::0",
        source_page=page,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
        strategy="fixed_512",
    )


# ── build_context ─────────────────────────────────────────────────────────────

def test_build_context_contains_source():
    chunk = make_chunk("Paris", "Paris is the capital of France.")
    context = build_context([(chunk, 0.9)])
    assert "Paris" in context


def test_build_context_multiple_chunks():
    chunks = [
        (make_chunk("Paris", "Paris text."), 0.9),
        (make_chunk("Tokyo", "Tokyo text."), 0.8),
    ]
    context = build_context(chunks)
    assert "Paris" in context
    assert "Tokyo" in context


def test_build_context_numbered():
    chunk = make_chunk("Paris")
    context = build_context([(chunk, 0.9)])
    assert "[1]" in context


# ── generate_answer ───────────────────────────────────────────────────────────

def test_generate_answer_calls_ollama():
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Paris is best visited in spring."}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        answer = generate_answer(
            question="When to visit Paris?",
            context="Paris context here.",
            model="llama3:8b",
        )

        assert answer == "Paris is best visited in spring."
        mock_post.assert_called_once()


def test_generate_answer_returns_string():
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "  Some answer.  "}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        answer = generate_answer("Q?", "Context.", model="llama3:8b")
        assert isinstance(answer, str)
        assert answer == "Some answer."


# ── RAGResponse ───────────────────────────────────────────────────────────────

def test_rag_response_fields():
    resp = RAGResponse(
        question="What to see in Paris?",
        answer="Visit the Eiffel Tower.",
        retrieved_chunks=[{"source_page": "Paris", "text": "...", "score": 0.9}],
        model="llama3:8b",
    )
    assert resp.question == "What to see in Paris?"
    assert resp.answer == "Visit the Eiffel Tower."
    assert len(resp.retrieved_chunks) == 1
    assert resp.model == "llama3:8b"
