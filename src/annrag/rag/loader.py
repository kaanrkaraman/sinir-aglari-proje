"""Article loader — reads raw JSON files produced by gt fetch / M2 pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_article(path: Path) -> tuple[str, str]:
    """Load a single article JSON.

    Returns:
        (page_title, text)
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    title: str = data["resolved_title"]
    text: str = data["extract"]
    return title, text


def load_all_articles(raw_dir: Path) -> list[tuple[str, str]]:
    """Load every *.json file in raw_dir.

    Returns:
        List of (page_title, text) tuples, sorted by title.
    """
    paths = sorted(raw_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No .json files found in {raw_dir}")

    articles = []
    for p in paths:
        title, text = load_article(p)
        articles.append((title, text))
        logger.debug("Loaded %s (%d chars)", title, len(text))

    logger.info("Loaded %d articles from %s", len(articles), raw_dir)
    return articles
