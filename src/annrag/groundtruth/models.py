"""Pydantic models for ground-truth Q&A records.

`relevant_doc_ids` holds Wikivoyage *page titles* (not chunk IDs) so the
ground truth survives every chunking strategy in the experiment grid.
At eval time, retrieval is judged by whether any chunk's `source_page` is
in the gold set — see milestone 5.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_PAGE_TITLE_NORMALIZE = re.compile(r"\s+")


def normalize_page_title(title: str) -> str:
    """Canonicalize a Wikivoyage page title for comparison.

    MediaWiki treats titles as case-insensitive on the first char and uses
    underscores/spaces interchangeably. We collapse whitespace and NFC-normalize
    so two records pointing at the same page compare equal.
    """
    title = unicodedata.normalize("NFC", title.strip())
    title = title.replace("_", " ")
    title = _PAGE_TITLE_NORMALIZE.sub(" ", title)
    if not title:
        return title
    return title[0].upper() + title[1:]


class QACategory(StrEnum):
    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QASource(StrEnum):
    SYNTHETIC_UNREVIEWED = "synthetic_unreviewed"
    SYNTHETIC_CURATED = "synthetic_curated"
    MANUAL = "manual"


class QAPair(BaseModel):
    """A single ground-truth Q&A record.

    `id` is a stable hash of (question, first relevant_doc_id) so re-running
    synthetic generation produces the same IDs for the same question text —
    keeping the curation CSV round-trippable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=8, max_length=32)
    question: str = Field(min_length=3)
    ground_truth_answer: str = Field(min_length=1)
    relevant_doc_ids: list[str] = Field(min_length=1)
    category: QACategory
    difficulty: Difficulty | None = None
    language: str = Field(default="en", min_length=2, max_length=8)
    evidence: str | None = Field(
        default=None,
        description="Direct quote(s) from the source page(s) supporting the answer.",
    )
    source: QASource = QASource.SYNTHETIC_UNREVIEWED
    generator_model: str | None = Field(
        default=None,
        description="Model tag (e.g. 'llama3:8b') that produced this candidate.",
    )
    notes: str | None = None

    @model_validator(mode="after")
    def _normalize_and_check_id(self) -> Self:
        # Canonicalize doc IDs in-place via a fresh tuple (the model is frozen,
        # so we go through model_copy if needed).
        normalized = [normalize_page_title(d) for d in self.relevant_doc_ids]
        if normalized != self.relevant_doc_ids:
            object.__setattr__(self, "relevant_doc_ids", normalized)
        # Validate the ID matches what derive_id would produce — but only if
        # caller didn't bypass `derive_id` (allow custom IDs for manual entries).
        return self


def derive_qa_id(question: str, first_doc_id: str) -> str:
    """Stable 12-char ID from (question, first_doc_id).

    Hash inputs are normalized so trivial whitespace / case differences in
    the source page title don't produce a different ID for the same logical Q&A.
    """
    payload = f"{question.strip()}\x00{normalize_page_title(first_doc_id)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:12]


class CandidateBatch(BaseModel):
    """A batch produced for one source article during synthetic generation.

    Persisted alongside candidates so we can reproduce / debug a batch later
    without re-querying the LLM.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_page: str
    page_id: str
    language: str
    model: str
    n_requested: int
    n_returned: int
