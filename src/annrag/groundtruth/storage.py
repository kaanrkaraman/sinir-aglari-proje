"""JSONL storage for QA records — streaming, deduplicated, schema-validated."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from annrag.groundtruth.models import QAPair

logger = logging.getLogger(__name__)


def iter_qa_pairs(path: Path) -> Iterator[QAPair]:
    """Yield validated QAPair records from a JSONL file. Skips blank lines."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                yield QAPair.model_validate_json(stripped)
            except ValueError as e:
                logger.error("skip malformed JSONL at %s:%d — %s", path, lineno, e)


def load_qa_pairs(path: Path) -> list[QAPair]:
    return list(iter_qa_pairs(path))


def save_qa_pairs(path: Path, records: Iterable[QAPair], *, dedupe: bool = True) -> int:
    """Write `records` to `path` as JSONL, returning the number written.

    With `dedupe=True` (default), repeated IDs collapse to the first occurrence —
    silently dropping later duplicates. The output dir is created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            if dedupe:
                if rec.id in seen:
                    continue
                seen.add(rec.id)
            fh.write(rec.model_dump_json())
            fh.write("\n")
            written += 1
    logger.info("wrote %d records to %s", written, path)
    return written


def append_qa_pairs(path: Path, records: Iterable[QAPair], *, dedupe: bool = True) -> int:
    """Append records to `path`, optionally deduplicating against existing IDs.

    Useful for incremental synthetic generation across multiple seed pages.
    """
    existing_ids: set[str] = (
        {r.id for r in iter_qa_pairs(path)} if dedupe and path.exists() else set()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            if dedupe and rec.id in existing_ids:
                continue
            existing_ids.add(rec.id)
            fh.write(rec.model_dump_json())
            fh.write("\n")
            written += 1
    logger.info("appended %d records to %s", written, path)
    return written


def load_seed_titles(path: Path) -> list[str]:
    """Load the curated seed-article list from a JSON file.

    Schema: `{"language": "en", "articles": ["Tokyo", "Paris", ...]}`.
    """
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"seed file {path} must be a JSON object")
    raw_articles = payload.get("articles")
    if not isinstance(raw_articles, list):
        raise ValueError(f"seed file {path} 'articles' must be a list[str]")
    articles: list[str] = []
    for a in raw_articles:
        if not isinstance(a, str):
            raise ValueError(f"seed file {path} 'articles' must be a list[str]")
        articles.append(a)
    return articles
