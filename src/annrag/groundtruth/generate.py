"""LLM-driven synthetic Q&A generation.

For each source article, force the model to call a single tool whose
input_schema is the validated shape we want back. The output is a list of
(question, answer, category, difficulty, evidence) records, which we wrap
into `QAPair` objects. Records are tagged `synthetic_unreviewed` until the
human curation step (`curate.py`) bumps them to `synthetic_curated`.

Multi-hop questions here are *intra-article* (combining two sections of the
same page). True cross-document multi-hop is out of scope for M2 and noted
as a limitation in the report.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from annrag.groundtruth.fetch import ArticleExtract
from annrag.groundtruth.models import (
    CandidateBatch,
    Difficulty,
    QACategory,
    QAPair,
    QASource,
    derive_qa_id,
)
from annrag.llm.base import LLMClient, LLMError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schema — the shape Claude is forced to return
# ---------------------------------------------------------------------------
QA_TOOL_NAME = "submit_qa_pairs"
QA_TOOL_DESCRIPTION = (
    "Submit the generated question/answer pairs for evaluation of a Wikivoyage RAG system."
)
QA_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "minLength": 5},
                    "answer": {"type": "string", "minLength": 1},
                    "category": {
                        "type": "string",
                        "enum": [c.value for c in QACategory],
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": [d.value for d in Difficulty],
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "Direct quote(s) from the article supporting the answer. "
                            "Use a single passage for factual; concatenate two passages "
                            "joined by ' ... ' for multi-hop."
                        ),
                    },
                },
                "required": ["question", "answer", "category", "difficulty", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa_pairs"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are creating ground-truth Q&A pairs for evaluating a travel-guide RAG system over Wikivoyage. The Q&A pairs you produce will be used to measure both retrieval quality and answer correctness, so they must be:

- Strictly answerable from the provided article (no outside knowledge).
- Concise — questions are one sentence; answers are one to three sentences.
- Diverse across the three categories defined below; aim for a roughly balanced mix.

Category definitions:
- factual:   Single piece of info from one section. Example: "What time does the Louvre open on Tuesdays?"
- multi_hop: Requires combining information from two sections of THIS article. Example: "Which neighborhood has both Michelin restaurants and is walkable from the central station?"
- ambiguous: Has multiple valid interpretations or would benefit from clarification. Example: "When is the best time to visit?" — best for what (weather, prices, crowds)?

Quality rules:
- Never invent facts the article does not state.
- Never produce yes/no questions for the factual category.
- Evidence MUST be a near-verbatim excerpt from the article. For multi_hop, concatenate the two supporting passages with ' ... ' between them.
- If the article is too short to support N pairs of the requested diversity, return fewer rather than padding with weak items."""


def make_user_prompt(article: ArticleExtract, n: int) -> str:
    return (
        f"Article title: {article.resolved_title}\n"
        f"Article language: {article.language}\n"
        f"Generate {n} Q&A pairs spanning all three categories.\n\n"
        f"--- ARTICLE TEXT ---\n{article.extract}\n--- END ARTICLE ---"
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class _RawQAItem(BaseModel):
    """Schema-mirror of one item the LLM returns; validated before wrapping in QAPair."""

    # `extra="ignore"` and the empty-string default for `evidence` are both
    # defensive against small local models that drift from the prompt: they
    # sometimes invent extra keys or omit required-by-prompt-but-not-by-schema
    # content like a verbatim quote.
    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=5)
    answer: str = Field(min_length=1)
    category: QACategory
    difficulty: Difficulty
    evidence: str = ""


class _RawQAResponse(BaseModel):
    """Top-level shape; items are validated individually so one bad row doesn't kill a batch."""

    model_config = ConfigDict(extra="ignore")

    qa_pairs: list[dict[str, Any]] = Field(min_length=1)


class QAGenerator:
    """Generates QA candidates for a single article via an LLM client."""

    def __init__(self, llm: LLMClient, *, model_name: str) -> None:
        self._llm = llm
        # `model_name` is informational (recorded in the batch); the LLM client
        # already knows which model it speaks to.
        self._model_name = model_name

    def generate(
        self,
        article: ArticleExtract,
        *,
        n: int,
    ) -> tuple[list[QAPair], CandidateBatch]:
        """Produce up to `n` candidates for `article`.

        Returns a (records, batch_metadata) tuple. Records are tagged with
        `synthetic_unreviewed` source so the curation step can transition them.
        Raises `LLMError` on unrecoverable LLM failure or schema mismatch.
        """
        user_prompt = make_user_prompt(article, n)
        try:
            raw = self._llm.call_tool(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                tool_name=QA_TOOL_NAME,
                tool_description=QA_TOOL_DESCRIPTION,
                input_schema=QA_TOOL_SCHEMA,
                max_tokens=4096,
            )
        except LLMError:
            raise
        try:
            parsed = _RawQAResponse.model_validate(raw)
        except ValidationError as e:
            raise LLMError(
                f"LLM returned malformed Q&A payload for {article.resolved_title!r}: {e}"
            ) from e

        records: list[QAPair] = []
        skipped = 0
        for raw_item in parsed.qa_pairs:
            try:
                item = _RawQAItem.model_validate(raw_item)
            except ValidationError as ve:
                skipped += 1
                logger.warning(
                    "skip malformed item from %s: %s",
                    article.resolved_title,
                    ve.errors()[0],
                )
                continue
            doc_id = article.resolved_title
            qa_id = derive_qa_id(item.question, doc_id)
            evidence = item.evidence.strip() or None
            records.append(
                QAPair(
                    id=qa_id,
                    question=item.question,
                    ground_truth_answer=item.answer,
                    relevant_doc_ids=[doc_id],
                    category=item.category,
                    difficulty=item.difficulty,
                    language=article.language,
                    evidence=evidence,
                    source=QASource.SYNTHETIC_UNREVIEWED,
                    generator_model=self._model_name,
                )
            )
        if skipped:
            logger.info(
                "skipped %d malformed items for page=%r (kept %d)",
                skipped,
                article.resolved_title,
                len(records),
            )
        if not records:
            raise LLMError(
                f"LLM produced no usable items for {article.resolved_title!r} (skipped={skipped})"
            )

        batch = CandidateBatch(
            source_page=article.resolved_title,
            page_id=article.page_id,
            language=article.language,
            model=self._model_name,
            n_requested=n,
            n_returned=len(records),
        )
        logger.info(
            "generated %d/%d candidates for page=%r (model=%s)",
            len(records),
            n,
            article.resolved_title,
            self._model_name,
        )
        return records, batch
