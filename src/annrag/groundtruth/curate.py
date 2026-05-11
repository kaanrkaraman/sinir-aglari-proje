"""CSV export/import for human-in-the-loop Q&A curation.

The CSV is the seam between LLM-generated candidates and the curated
ground-truth dataset. Reviewers open it in any spreadsheet, set
`accepted=y` on the rows to keep (optionally editing the question or
answer text), and `import_curated` re-derives stable IDs for any edits
and bumps the `source` enum to `synthetic_curated`.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from pathlib import Path

from annrag.groundtruth.models import (
    Difficulty,
    QACategory,
    QAPair,
    QASource,
    derive_qa_id,
)
from annrag.groundtruth.storage import save_qa_pairs

logger = logging.getLogger(__name__)

CSV_FIELDS: list[str] = [
    "id",
    "accepted",
    "question",
    "ground_truth_answer",
    "category",
    "difficulty",
    "language",
    "relevant_doc_ids",
    "evidence",
    "generator_model",
    "notes",
]

_DOC_IDS_SEP = " | "
_ACCEPTED_TRUTHY = {"y", "yes", "true", "1", "accept", "accepted"}


def export_for_review(records: Iterable[QAPair], path: Path) -> int:
    """Write `records` to `path` as a CSV with an empty `accepted` column.

    Returns the number of rows written. Reviewers fill `accepted` with
    y/yes/true on rows they want to keep; `import_curated` reads it back.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "id": rec.id,
                    "accepted": "",
                    "question": rec.question,
                    "ground_truth_answer": rec.ground_truth_answer,
                    "category": rec.category.value,
                    "difficulty": rec.difficulty.value if rec.difficulty else "",
                    "language": rec.language,
                    "relevant_doc_ids": _DOC_IDS_SEP.join(rec.relevant_doc_ids),
                    "evidence": rec.evidence or "",
                    "generator_model": rec.generator_model or "",
                    "notes": rec.notes or "",
                }
            )
            written += 1
    logger.info("exported %d rows to %s", written, path)
    return written


def import_curated(csv_path: Path, out_path: Path) -> int:
    """Read `csv_path`, keep accepted rows, write them to `out_path` as JSONL.

    Re-derives `id` from the (possibly edited) question + first doc id so the
    output dataset has stable, content-derived IDs even if the reviewer edited
    a question. The `source` is bumped to `SYNTHETIC_CURATED` to record that
    a human signed off.
    Returns the number of records written.
    """
    accepted: list[QAPair] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        missing = [c for c in CSV_FIELDS if c not in fieldnames]
        if missing:
            raise ValueError(
                f"{csv_path}: missing required columns: {missing}"
            )
        for row in reader:
            if not _is_accepted(row.get("accepted")):
                continue
            accepted.append(_row_to_qa(row))
    return save_qa_pairs(out_path, accepted)


def _is_accepted(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in _ACCEPTED_TRUTHY


def _row_to_qa(row: dict[str, str]) -> QAPair:
    question = (row.get("question") or "").strip()
    doc_ids = [
        d.strip()
        for d in (row.get("relevant_doc_ids") or "").split("|")
        if d.strip()
    ]
    if not doc_ids:
        raise ValueError(
            f"accepted row {row.get('id')!r} has empty relevant_doc_ids"
        )
    diff_raw = (row.get("difficulty") or "").strip()
    difficulty = Difficulty(diff_raw) if diff_raw else None
    language = (row.get("language") or "").strip() or "en"
    return QAPair(
        id=derive_qa_id(question, doc_ids[0]),
        question=question,
        ground_truth_answer=(row.get("ground_truth_answer") or "").strip(),
        relevant_doc_ids=doc_ids,
        category=QACategory((row.get("category") or "").strip()),
        difficulty=difficulty,
        language=language,
        evidence=(row.get("evidence") or "").strip() or None,
        source=QASource.SYNTHETIC_CURATED,
        generator_model=(row.get("generator_model") or "").strip() or None,
        notes=(row.get("notes") or "").strip() or None,
    )
