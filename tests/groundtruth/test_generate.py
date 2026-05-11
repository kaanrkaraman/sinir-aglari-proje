"""Unit tests for the QA generator with a fake LLMClient."""

from __future__ import annotations

from typing import Any

import pytest

from annrag.groundtruth.fetch import ArticleExtract
from annrag.groundtruth.generate import (
    QA_TOOL_NAME,
    QA_TOOL_SCHEMA,
    SYSTEM_PROMPT,
    QAGenerator,
)
from annrag.groundtruth.models import QACategory, QASource
from annrag.llm.base import LLMError


class FakeLLM:
    def __init__(self, payload, capture: dict | None = None):
        self._payload = payload
        self._capture = capture if capture is not None else {}

    def call_tool(self, **kwargs) -> dict[str, Any]:
        self._capture.update(kwargs)
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _article(extract="Tokyo is huge. The Shibuya crossing is famous.", title="Tokyo"):
    return ArticleExtract(
        requested_title=title,
        resolved_title=title,
        page_id="1",
        language="en",
        extract=extract,
    )


class TestGenerate:
    def test_happy_path_wraps_records(self):
        capture: dict = {}
        payload = {
            "qa_pairs": [
                {
                    "question": "What is Tokyo's most famous crossing?",
                    "answer": "The Shibuya crossing.",
                    "category": "factual",
                    "difficulty": "easy",
                    "evidence": "The Shibuya crossing is famous.",
                },
                {
                    "question": "How would you summarize Tokyo's scale?",
                    "answer": "It is a huge city.",
                    "category": "ambiguous",
                    "difficulty": "medium",
                    "evidence": "Tokyo is huge.",
                },
            ]
        }
        gen = QAGenerator(FakeLLM(payload, capture), model_name="claude-test-1")
        records, batch = gen.generate(_article(), n=2)

        assert len(records) == 2
        assert records[0].category is QACategory.FACTUAL
        assert records[0].source is QASource.SYNTHETIC_UNREVIEWED
        assert records[0].relevant_doc_ids == ["Tokyo"]
        assert batch.source_page == "Tokyo"
        assert batch.n_requested == 2
        assert batch.n_returned == 2
        assert batch.model == "claude-test-1"

        # The LLM was called with the right tool descriptor and a system prompt.
        assert capture["tool_name"] == QA_TOOL_NAME
        assert capture["input_schema"] is QA_TOOL_SCHEMA
        assert SYSTEM_PROMPT.startswith("You are creating ground-truth")
        assert capture["system"] == SYSTEM_PROMPT
        assert "Tokyo is huge" in capture["user"]

    def test_invalid_payload_raises_llmerror(self):
        bad_payload = {"qa_pairs": [{"question": "?", "answer": "x"}]}  # missing fields
        gen = QAGenerator(FakeLLM(bad_payload), model_name="m")
        with pytest.raises(LLMError):
            gen.generate(_article(), n=1)

    def test_empty_payload_raises_llmerror(self):
        gen = QAGenerator(FakeLLM({"qa_pairs": []}), model_name="m")
        with pytest.raises(LLMError):
            gen.generate(_article(), n=1)

    def test_llm_error_propagates(self):
        gen = QAGenerator(FakeLLM(LLMError("boom")), model_name="m")
        with pytest.raises(LLMError, match="boom"):
            gen.generate(_article(), n=1)
