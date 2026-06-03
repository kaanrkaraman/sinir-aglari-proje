"""End-to-end RAG pipeline — M6.

Retrieves relevant chunks from FAISS index and generates
an answer using llama3:8b via Ollama.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"

SYSTEM_PROMPT = """\
You are a helpful travel guide assistant. Answer the user's question
based ONLY on the provided context passages from Wikivoyage.

Rules:
- Use only information from the context passages below.
- If the context does not contain enough information, say so clearly.
- Be concise and specific.
- Cite which city/page the information comes from when relevant.
"""


@dataclass
class RAGResponse:
    """Result of a single RAG query."""

    question: str
    answer: str
    retrieved_chunks: list[dict]   # [{source_page, text, score}]
    model: str


def build_context(chunks: list[tuple]) -> str:
    """Build context string from retrieved chunks."""
    parts = []
    for i, (chunk, score) in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk.source_page}\n"
            f"{chunk.text[:500]}"
        )
    return "\n\n".join(parts)


def generate_answer(
    question: str,
    context: str,
    model: str = "llama3:8b",
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """Call Ollama chat API to generate an answer."""
    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer based only on the context above:"
    )

    client = httpx.Client(timeout=120.0)
    resp = client.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        },
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def rag_ask(
    question: str,
    index_dir: str,
    embedder_model: str = "nomic-embed-text",
    llm_model: str = "llama3:8b",
    top_k: int = 5,
) -> RAGResponse:
    """Full RAG pipeline: embed → retrieve → generate.

    Args:
        question: User question.
        index_dir: Path to FAISS index directory.
        embedder_model: Ollama embedding model.
        llm_model: Ollama LLM model for answer generation.
        top_k: Number of chunks to retrieve.

    Returns:
        RAGResponse with answer and retrieved chunks.
    """
    from pathlib import Path
    from annrag.rag.embedder import OllamaEmbedder
    from annrag.rag.index import VectorIndex

    # Load index
    index = VectorIndex.load(Path(index_dir))
    embedder = OllamaEmbedder(model=embedder_model)

    # Embed question
    logger.info("Embedding question...")
    query_vec = embedder.embed(question)

    # Retrieve top-k chunks
    logger.info("Retrieving top-%d chunks...", top_k)
    results = index.search(query_vec, top_k=top_k)

    # Build context
    context = build_context(results)

    # Generate answer
    logger.info("Generating answer with %s...", llm_model)
    answer = generate_answer(question, context, model=llm_model)

    return RAGResponse(
        question=question,
        answer=answer,
        retrieved_chunks=[
            {
                "source_page": chunk.source_page,
                "text": chunk.text[:200],
                "score": round(score, 4),
            }
            for chunk, score in results
        ],
        model=llm_model,
    )
