"""Distribution stats over a Q&A dataset — for the report and sanity checks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import NamedTuple

from annrag.groundtruth.models import QAPair


class DatasetStats(NamedTuple):
    total: int
    by_category: dict[str, int]
    by_difficulty: dict[str, int]
    by_source: dict[str, int]
    by_language: dict[str, int]
    by_page: dict[str, int]
    pages_count: int
    avg_question_chars: float
    avg_answer_chars: float

    def render(self) -> str:
        """One-line-per-row human-readable summary."""
        lines = [
            f"total records:       {self.total}",
            f"unique source pages: {self.pages_count}",
            f"avg question chars:  {self.avg_question_chars:.1f}",
            f"avg answer   chars:  {self.avg_answer_chars:.1f}",
            "",
            "by category:",
            *_render_dict(self.by_category),
            "",
            "by difficulty:",
            *_render_dict(self.by_difficulty),
            "",
            "by source:",
            *_render_dict(self.by_source),
            "",
            "by language:",
            *_render_dict(self.by_language),
            "",
            f"per-page record counts (top 10 of {self.pages_count}):",
            *_render_dict(dict(sorted(self.by_page.items(), key=lambda kv: -kv[1])[:10])),
        ]
        return "\n".join(lines)


def _render_dict(d: dict[str, int]) -> list[str]:
    if not d:
        return ["  (none)"]
    width = max(len(k) for k in d)
    return [f"  {k:<{width}}  {v}" for k, v in d.items()]


def compute_stats(records: Iterable[QAPair]) -> DatasetStats:
    cat: Counter[str] = Counter()
    diff: Counter[str] = Counter()
    src: Counter[str] = Counter()
    lang: Counter[str] = Counter()
    page: Counter[str] = Counter()
    total = 0
    q_chars = 0
    a_chars = 0
    for r in records:
        total += 1
        cat[r.category.value] += 1
        diff[(r.difficulty.value if r.difficulty else "unspecified")] += 1
        src[r.source.value] += 1
        lang[r.language] += 1
        for d in r.relevant_doc_ids:
            page[d] += 1
        q_chars += len(r.question)
        a_chars += len(r.ground_truth_answer)
    return DatasetStats(
        total=total,
        by_category=dict(cat),
        by_difficulty=dict(diff),
        by_source=dict(src),
        by_language=dict(lang),
        by_page=dict(page),
        pages_count=len(page),
        avg_question_chars=(q_chars / total) if total else 0.0,
        avg_answer_chars=(a_chars / total) if total else 0.0,
    )
