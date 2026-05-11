"""Unit tests for JSONL storage."""

from __future__ import annotations

import json

import pytest

from annrag.groundtruth.models import Difficulty, QACategory, QAPair, derive_qa_id
from annrag.groundtruth.storage import (
    append_qa_pairs,
    iter_qa_pairs,
    load_qa_pairs,
    load_seed_titles,
    save_qa_pairs,
)


def _qa(question, page="Tokyo"):
    return QAPair(
        id=derive_qa_id(question, page),
        question=question,
        ground_truth_answer="x",
        relevant_doc_ids=[page],
        category=QACategory.FACTUAL,
        difficulty=Difficulty.EASY,
        evidence="x",
    )


class TestRoundTrip:
    def test_save_and_load(self, tmp_path):
        out = tmp_path / "qa.jsonl"
        records = [_qa("Q1?"), _qa("Q2?"), _qa("Q3?")]
        save_qa_pairs(out, records)
        loaded = load_qa_pairs(out)
        assert [r.id for r in loaded] == [r.id for r in records]


class TestDeduplication:
    def test_save_dedupes(self, tmp_path):
        out = tmp_path / "qa.jsonl"
        rec = _qa("Q1?")
        save_qa_pairs(out, [rec, rec, rec])
        assert len(load_qa_pairs(out)) == 1

    def test_save_keeps_dupes_when_disabled(self, tmp_path):
        out = tmp_path / "qa.jsonl"
        rec = _qa("Q1?")
        save_qa_pairs(out, [rec, rec], dedupe=False)
        assert len(load_qa_pairs(out)) == 2


class TestAppend:
    def test_append_dedupes_against_existing(self, tmp_path):
        out = tmp_path / "qa.jsonl"
        save_qa_pairs(out, [_qa("Q1?")])
        appended = append_qa_pairs(out, [_qa("Q1?"), _qa("Q2?")])
        assert appended == 1
        assert {r.question for r in load_qa_pairs(out)} == {"Q1?", "Q2?"}


class TestMalformed:
    def test_skips_bad_lines(self, tmp_path):
        out = tmp_path / "qa.jsonl"
        good = _qa("Q1?")
        out.write_text(
            good.model_dump_json()
            + "\n"
            + "not-json\n"
            + '{"id":"too-short"}\n'  # invalid (will fail validation)
            + good.model_dump_json()
            + "\n",  # duplicate of good — caller handles dedupe at write time, not on read
            encoding="utf-8",
        )
        loaded = list(iter_qa_pairs(out))
        # Two valid records, malformed lines dropped silently (logged).
        assert len(loaded) == 2

    def test_missing_file_yields_empty(self, tmp_path):
        assert list(iter_qa_pairs(tmp_path / "nope.jsonl")) == []


class TestSeedLoader:
    def test_loads_articles_list(self, tmp_path):
        seed = tmp_path / "seed.json"
        seed.write_text(json.dumps({"articles": ["Tokyo", "Paris"]}), encoding="utf-8")
        assert load_seed_titles(seed) == ["Tokyo", "Paris"]

    def test_rejects_non_object(self, tmp_path):
        seed = tmp_path / "seed.json"
        seed.write_text(json.dumps(["Tokyo"]), encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_seed_titles(seed)

    def test_rejects_missing_articles(self, tmp_path):
        seed = tmp_path / "seed.json"
        seed.write_text(json.dumps({"articles": [1, 2, 3]}), encoding="utf-8")
        with pytest.raises(ValueError, match="list\\[str\\]"):
            load_seed_titles(seed)
