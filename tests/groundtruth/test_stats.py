"""Stats reporter tests."""

from __future__ import annotations

from annrag.groundtruth.models import (
    Difficulty,
    QACategory,
    QAPair,
    QASource,
    derive_qa_id,
)
from annrag.groundtruth.stats import compute_stats


def _qa(question, page, category, difficulty=Difficulty.EASY, source=QASource.SYNTHETIC_UNREVIEWED):
    return QAPair(
        id=derive_qa_id(question, page),
        question=question,
        ground_truth_answer="ans",
        relevant_doc_ids=[page],
        category=category,
        difficulty=difficulty,
        evidence="evi",
        source=source,
    )


def test_compute_stats_counts():
    records = [
        _qa("Q1?", "Tokyo", QACategory.FACTUAL),
        _qa("Q2?", "Tokyo", QACategory.MULTI_HOP, difficulty=Difficulty.HARD),
        _qa("Q3?", "Paris", QACategory.AMBIGUOUS, source=QASource.SYNTHETIC_CURATED),
    ]
    s = compute_stats(records)
    assert s.total == 3
    assert s.pages_count == 2
    assert s.by_category == {"factual": 1, "multi_hop": 1, "ambiguous": 1}
    assert s.by_difficulty == {"easy": 2, "hard": 1}
    assert s.by_source == {"synthetic_unreviewed": 2, "synthetic_curated": 1}
    assert s.by_page == {"Tokyo": 2, "Paris": 1}
    assert s.by_language == {"en": 3}


def test_render_does_not_crash_on_empty():
    s = compute_stats([])
    out = s.render()
    assert "total records:       0" in out
