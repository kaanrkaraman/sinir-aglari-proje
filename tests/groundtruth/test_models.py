"""Unit tests for ground-truth Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from annrag.groundtruth.models import (
    Difficulty,
    QACategory,
    QAPair,
    QASource,
    derive_qa_id,
    normalize_page_title,
)


class TestNormalizePageTitle:
    def test_underscore_to_space(self):
        assert normalize_page_title("New_York_City") == "New York City"

    def test_collapses_whitespace(self):
        assert normalize_page_title("  Tokyo   Station  ") == "Tokyo Station"

    def test_uppercases_first_letter(self):
        assert normalize_page_title("paris") == "Paris"

    def test_preserves_internal_case(self):
        assert normalize_page_title("McMurdo Station") == "McMurdo Station"

    def test_empty_string(self):
        assert normalize_page_title("") == ""


class TestDeriveQaId:
    def test_stable_for_same_inputs(self):
        a = derive_qa_id("What time?", "Tokyo")
        b = derive_qa_id("What time?", "Tokyo")
        assert a == b
        assert len(a) == 12

    def test_changes_with_question(self):
        assert derive_qa_id("a?", "Tokyo") != derive_qa_id("b?", "Tokyo")

    def test_normalizes_doc_id(self):
        # Underscore vs space in the page title produces the same ID.
        assert derive_qa_id("q?", "Tokyo_Station") == derive_qa_id("q?", "Tokyo Station")


class TestQAPair:
    def _kwargs(self, **overrides):
        defaults = {
            "id": "abc123def456",
            "question": "What is the capital of France?",
            "ground_truth_answer": "Paris.",
            "relevant_doc_ids": ["Paris"],
            "category": QACategory.FACTUAL,
            "difficulty": Difficulty.EASY,
            "evidence": "Paris is the capital of France.",
        }
        defaults.update(overrides)
        return defaults

    def test_construct_minimal(self):
        rec = QAPair(**self._kwargs())
        assert rec.source is QASource.SYNTHETIC_UNREVIEWED
        assert rec.language == "en"

    def test_normalizes_doc_ids(self):
        rec = QAPair(**self._kwargs(relevant_doc_ids=["new_york_city", "paris"]))
        assert rec.relevant_doc_ids == ["New york city", "Paris"]

    def test_rejects_empty_doc_list(self):
        with pytest.raises(ValidationError):
            QAPair(**self._kwargs(relevant_doc_ids=[]))

    def test_rejects_unknown_category(self):
        kwargs = self._kwargs()
        kwargs["category"] = "trivia"
        with pytest.raises(ValidationError):
            QAPair(**kwargs)

    def test_frozen(self):
        rec = QAPair(**self._kwargs())
        with pytest.raises(ValidationError):
            rec.question = "changed"  # type: ignore[misc]

    def test_extra_fields_forbidden(self):
        kwargs = self._kwargs()
        kwargs["unknown"] = "x"
        with pytest.raises(ValidationError):
            QAPair(**kwargs)
