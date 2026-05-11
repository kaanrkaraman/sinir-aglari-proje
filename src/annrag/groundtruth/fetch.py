"""Wikivoyage MediaWiki API client.

We use `prop=extracts&explaintext` to pull plain-text article bodies — that
form is well-suited for sending to an LLM for synthetic Q&A generation. Full
wikitext (templates, tables, etc.) is left for milestone 3 ingestion which
will use the XML dump for the indexed corpus.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ArticleExtract(BaseModel):
    """One article fetched from the MediaWiki API."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requested_title: str
    resolved_title: str = Field(description="Title after redirects (e.g. 'NYC' → 'New York City').")
    page_id: str
    language: str
    extract: str = Field(description="Plain-text article body, sections separated by blank lines.")


class WikivoyageFetchError(RuntimeError):
    """Raised when the API returns an unexpected payload."""


class WikivoyageFetcher:
    """Polite, throttled client for fetching plain-text Wikivoyage extracts."""

    def __init__(
        self,
        *,
        lang: str,
        user_agent: str,
        delay_s: float,
        client: httpx.Client | None = None,
    ) -> None:
        self._lang = lang
        self._base_url = f"https://{lang}.wikivoyage.org/w/api.php"
        self._user_agent = user_agent
        self._delay_s = delay_s
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
        self._last_call_monotonic: float = 0.0

    def __enter__(self) -> WikivoyageFetcher:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch_extract(self, title: str) -> ArticleExtract | None:
        """Fetch one article. Returns None if the page is missing/redirect-broken."""
        self._respect_delay()
        params: dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "redirects": 1,
            "titles": title,
        }
        response = self._client.get(
            self._base_url,
            params=params,
            headers={"User-Agent": self._user_agent, "Accept": "application/json"},
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return self._parse_extract_response(title, payload)

    def _parse_extract_response(
        self, requested_title: str, payload: dict[str, Any]
    ) -> ArticleExtract | None:
        try:
            pages = payload["query"]["pages"]
        except (KeyError, TypeError) as e:
            raise WikivoyageFetchError(
                f"unexpected payload shape for {requested_title!r}: {payload!r}"
            ) from e
        if not pages:
            return None
        page = pages[0] if isinstance(pages, list) else next(iter(pages.values()))
        if page.get("missing"):
            logger.warning("page missing on Wikivoyage: %r", requested_title)
            return None
        extract = page.get("extract") or ""
        if not extract.strip():
            logger.warning("empty extract for %r", requested_title)
            return None
        return ArticleExtract(
            requested_title=requested_title,
            resolved_title=page["title"],
            page_id=str(page["pageid"]),
            language=self._lang,
            extract=extract,
        )

    def _respect_delay(self) -> None:
        if self._delay_s <= 0:
            return
        now = time.monotonic()
        gap = now - self._last_call_monotonic
        wait = self._delay_s - gap
        if wait > 0:
            time.sleep(wait)
        self._last_call_monotonic = time.monotonic()
