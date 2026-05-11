"""CSV curation round-trip tests."""

from __future__ import annotations

import pytest

from annrag.groundtruth.curate import (
    CSV_FIELDS,
    export_for_review,
    import_curated,
)
from annrag.groundtruth.models import (
    Difficulty,
    QACategory,
    QAPair,
    QASource,
    derive_qa_id,
)
from annrag.groundtruth.storage import load_qa_pairs


def _qa(question, page="Tokyo", category=QACategory.FACTUAL):
    return QAPair(
        id=derive_qa_id(question, page),
        question=question,
        ground_truth_answer="x",
        relevant_doc_ids=[page],
        category=category,
        difficulty=Difficulty.EASY,
        evidence="x",
    )


class TestExport:
    def test_writes_header_and_rows(self, tmp_path):
        csv_path = tmp_path / "review.csv"
        export_for_review([_qa("Q1?"), _qa("Q2?")], csv_path)
        text = csv_path.read_text(encoding="utf-8")
        first_line = text.splitlines()[0]
        for col in CSV_FIELDS:
            assert col in first_line
        assert "Q1?" in text
        assert "Q2?" in text


class TestImport:
    def test_keeps_only_accepted(self, tmp_path):
        csv_path = tmp_path / "review.csv"
        out_path = tmp_path / "final.jsonl"
        export_for_review([_qa("Q1?"), _qa("Q2?"), _qa("Q3?")], csv_path)

        # Mark only first and third rows accepted.
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        header, *rows = lines
        rows[0] = rows[0].replace(",,", ",y,", 1)  # accepted column is 2nd field
        rows[2] = rows[2].replace(",,", ",true,", 1)
        csv_path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

        n = import_curated(csv_path, out_path)
        assert n == 2
        loaded = load_qa_pairs(out_path)
        assert {r.question for r in loaded} == {"Q1?", "Q3?"}
        assert all(r.source is QASource.SYNTHETIC_CURATED for r in loaded)

    def test_edits_to_question_re_derive_id(self, tmp_path):
        csv_path = tmp_path / "review.csv"
        out_path = tmp_path / "final.jsonl"
        export_for_review([_qa("original?")], csv_path)

        # Edit the question text and accept.
        text = csv_path.read_text(encoding="utf-8")
        edited = text.replace("original?", "edited?").replace(",,", ",y,", 1)
        csv_path.write_text(edited, encoding="utf-8")

        import_curated(csv_path, out_path)
        loaded = load_qa_pairs(out_path)
        assert loaded[0].question == "edited?"
        assert loaded[0].id == derive_qa_id("edited?", "Tokyo")

    def test_missing_columns_rejected(self, tmp_path):
        csv_path = tmp_path / "broken.csv"
        csv_path.write_text("id,question\nabc,foo\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            import_curated(csv_path, tmp_path / "out.jsonl")
