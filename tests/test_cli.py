"""CLI smoke tests via Typer's CliRunner."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from annrag._cli import app
from annrag.groundtruth.models import (
    Difficulty,
    QACategory,
    QAPair,
    derive_qa_id,
)
from annrag.groundtruth.storage import save_qa_pairs

runner = CliRunner()


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


def test_bootstrap_runs():
    result = runner.invoke(app, ["bootstrap"])
    assert result.exit_code == 0, result.output


def test_gt_help():
    result = runner.invoke(app, ["gt", "--help"])
    assert result.exit_code == 0
    assert "fetch" in result.output
    assert "generate" in result.output
    assert "stats" in result.output


def test_gt_stats_json(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNRAG_DATA_DIR", str(tmp_path))
    jsonl = tmp_path / "qa.jsonl"
    save_qa_pairs(jsonl, [_qa("Q1?"), _qa("Q2?")])
    result = runner.invoke(app, ["gt", "stats", str(jsonl), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total"] == 2
    assert payload["by_category"]["factual"] == 2


def test_gt_export_then_import_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNRAG_DATA_DIR", str(tmp_path))
    candidates = tmp_path / "candidates.jsonl"
    review = tmp_path / "review.csv"
    final = tmp_path / "final.jsonl"
    save_qa_pairs(candidates, [_qa("Q1?"), _qa("Q2?")])

    r1 = runner.invoke(app, ["gt", "export", "--in", str(candidates), "--out", str(review)])
    assert r1.exit_code == 0, r1.output

    # Mark both rows accepted.
    text = review.read_text(encoding="utf-8")
    review.write_text(text.replace(",,", ",y,"), encoding="utf-8")

    r2 = runner.invoke(app, ["gt", "import", "--in", str(review), "--out", str(final)])
    assert r2.exit_code == 0, r2.output
    assert final.exists()


def test_gt_generate_anthropic_without_key_fails(tmp_path, monkeypatch):
    """Default provider is Ollama, so verify Anthropic gating still works on override."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANNRAG_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANNRAG_DATA_DIR", str(tmp_path))
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    result = runner.invoke(
        app, ["gt", "generate", "--provider", "anthropic", "--raw-dir", str(raw_dir)]
    )
    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output
