"""Tests for M5 retrieval metrics."""

from __future__ import annotations

import math

from annrag.rag.metrics import QueryResult, RetrievalMetrics


def make_result(
    retrieved: list[str],
    relevant: list[str] = None,
) -> QueryResult:
    if relevant is None:
        relevant = ["Paris"]
    return QueryResult(
        question="Test question?",
        relevant_pages=relevant,
        retrieved_pages=retrieved,
        retrieved_scores=[1.0 - i * 0.1 for i in range(len(retrieved))],
    )


# ── Reciprocal Rank ───────────────────────────────────────────────────────────

def test_reciprocal_rank_first():
    r = make_result(["Paris", "London", "Tokyo"])
    assert r.reciprocal_rank == 1.0


def test_reciprocal_rank_second():
    r = make_result(["London", "Paris", "Tokyo"])
    assert r.reciprocal_rank == 0.5


def test_reciprocal_rank_not_found():
    r = make_result(["London", "Tokyo", "Berlin"])
    assert r.reciprocal_rank == 0.0


# ── Recall@k ─────────────────────────────────────────────────────────────────

def test_recall_at_1_found():
    r = make_result(["Paris", "London", "Tokyo"])
    assert r.recall_at_k(1) == 1.0


def test_recall_at_1_not_found():
    r = make_result(["London", "Paris", "Tokyo"])
    assert r.recall_at_k(1) == 0.0


def test_recall_at_3_found():
    r = make_result(["London", "Tokyo", "Paris"])
    assert r.recall_at_k(3) == 1.0


def test_recall_at_5_not_found():
    r = make_result(["London", "Tokyo", "Berlin", "Rome", "NYC"])
    assert r.recall_at_k(5) == 0.0


# ── NDCG@k ───────────────────────────────────────────────────────────────────

def test_ndcg_perfect():
    r = make_result(["Paris", "London", "Tokyo"])
    assert r.ndcg_at_k(5) == 1.0


def test_ndcg_lower_rank():
    r1 = make_result(["Paris", "London", "Tokyo"])
    r2 = make_result(["London", "Tokyo", "Paris"])
    assert r1.ndcg_at_k(5) > r2.ndcg_at_k(5)


def test_ndcg_not_found():
    r = make_result(["London", "Tokyo", "Berlin"])
    assert r.ndcg_at_k(5) == 0.0


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def test_mrr_perfect():
    results = [
        make_result(["Paris", "London"]),
        make_result(["Paris", "Tokyo"]),
    ]
    m = RetrievalMetrics(query_results=results)
    assert m.mrr == 1.0


def test_mrr_mixed():
    results = [
        make_result(["Paris", "London"]),   # RR = 1.0
        make_result(["London", "Paris"]),   # RR = 0.5
    ]
    m = RetrievalMetrics(query_results=results)
    assert abs(m.mrr - 0.75) < 1e-6


def test_recall_aggregate():
    results = [
        make_result(["Paris", "London"]),
        make_result(["London", "Tokyo", "Paris"]),
    ]
    m = RetrievalMetrics(query_results=results)
    assert m.recall_at_k(3) == 1.0


def test_render_contains_mrr():
    m = RetrievalMetrics(query_results=[make_result(["Paris"])])
    report = m.render()
    assert "MRR" in report


def test_to_dict_keys():
    m = RetrievalMetrics(query_results=[make_result(["Paris"])])
    d = m.to_dict()
    assert "mrr" in d
    assert "recall" in d
    assert "ndcg" in d
