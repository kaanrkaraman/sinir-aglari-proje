"""Retrieval metrics — M5.

Evaluates retrieval quality using ground truth Q&A pairs.
Metrics: MRR, Recall@k, NDCG@k
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Per-query result ──────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Retrieval result for a single query."""

    question: str
    relevant_pages: list[str]       # Gold pages from ground truth
    retrieved_pages: list[str]      # Pages from top-k chunks
    retrieved_scores: list[float]   # Similarity scores

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of first relevant page. 0 if not found."""
        for i, page in enumerate(self.retrieved_pages):
            if page in self.relevant_pages:
                return 1.0 / (i + 1)
        return 0.0

    def recall_at_k(self, k: int) -> float:
        """Fraction of relevant pages found in top-k."""
        top_k = set(self.retrieved_pages[:k])
        relevant = set(self.relevant_pages)
        if not relevant:
            return 0.0
        return len(top_k & relevant) / len(relevant)

    def ndcg_at_k(self, k: int) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        def dcg(pages: list[str], k: int) -> float:
            score = 0.0
            for i, page in enumerate(pages[:k]):
                if page in self.relevant_pages:
                    score += 1.0 / math.log2(i + 2)
            return score

        actual = dcg(self.retrieved_pages, k)
        # Ideal: all relevant pages at top
        ideal_pages = self.relevant_pages + [""] * k
        ideal = dcg(ideal_pages, k)
        return actual / ideal if ideal > 0 else 0.0


# ── Aggregate metrics ─────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    """Aggregate retrieval metrics over all queries."""

    query_results: list[QueryResult] = field(default_factory=list)
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])

    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank."""
        if not self.query_results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.query_results) / len(self.query_results)

    def recall_at_k(self, k: int) -> float:
        """Mean Recall@k."""
        if not self.query_results:
            return 0.0
        return sum(r.recall_at_k(k) for r in self.query_results) / len(self.query_results)

    def ndcg_at_k(self, k: int) -> float:
        """Mean NDCG@k."""
        if not self.query_results:
            return 0.0
        return sum(r.ndcg_at_k(k) for r in self.query_results) / len(self.query_results)

    def render(self) -> str:
        """Human-readable metrics report."""
        lines = [
            f"Queries evaluated: {len(self.query_results)}",
            f"MRR:              {self.mrr:.4f}",
        ]
        for k in self.k_values:
            lines.append(f"Recall@{k}:        {self.recall_at_k(k):.4f}")
        for k in self.k_values:
            lines.append(f"NDCG@{k}:          {self.ndcg_at_k(k):.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize metrics to dict for JSON output."""
        return {
            "n_queries": len(self.query_results),
            "mrr": round(self.mrr, 4),
            "recall": {f"@{k}": round(self.recall_at_k(k), 4) for k in self.k_values},
            "ndcg": {f"@{k}": round(self.ndcg_at_k(k), 4) for k in self.k_values},
        }


# ── Evaluator ─────────────────────────────────────────────────────────────────

def evaluate_retrieval(
    final_jsonl: Path,
    index_dir: Path,
    embedder_model: str = "nomic-embed-text",
    top_k: int = 10,
) -> RetrievalMetrics:
    """Run retrieval evaluation over all ground truth Q&A pairs.

    Args:
        final_jsonl: Path to final.jsonl (ground truth).
        index_dir: Path to FAISS index directory.
        embedder_model: Ollama embedding model name.
        top_k: Number of chunks to retrieve per query.

    Returns:
        RetrievalMetrics with all query results.
    """
    from annrag.rag.embedder import OllamaEmbedder
    from annrag.rag.index import VectorIndex

    # Load index
    logger.info("Loading index from %s", index_dir)
    index = VectorIndex.load(index_dir)
    embedder = OllamaEmbedder(model=embedder_model)

    # Load ground truth
    qa_pairs = []
    with open(final_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))

    logger.info("Evaluating %d queries", len(qa_pairs))

    metrics = RetrievalMetrics()

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        relevant_pages = qa.get("relevant_doc_ids", [])

        # Embed query
        query_vec = embedder.embed(question)

        # Retrieve top-k chunks
        results = index.search(query_vec, top_k=top_k)

        retrieved_pages = [chunk.source_page for chunk, _ in results]
        retrieved_scores = [score for _, score in results]

        query_result = QueryResult(
            question=question,
            relevant_pages=relevant_pages,
            retrieved_pages=retrieved_pages,
            retrieved_scores=retrieved_scores,
        )
        metrics.query_results.append(query_result)

        if (i + 1) % 10 == 0:
            logger.info("Evaluated %d / %d queries", i + 1, len(qa_pairs))

    return metrics
