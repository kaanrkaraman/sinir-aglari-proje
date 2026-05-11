"""Unit tests for the Wikivoyage fetcher — uses httpx.MockTransport, no network."""

from __future__ import annotations

import httpx
import pytest

from annrag.groundtruth.fetch import (
    ArticleExtract,
    WikivoyageFetcher,
    WikivoyageFetchError,
)


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _build_response(*, title="Tokyo", page_id=12345, extract="Tokyo is the capital."):
    """Mimic the formatversion=2 response shape."""
    return {"query": {"pages": [{"title": title, "pageid": page_id, "extract": extract}]}}


class TestFetchExtract:
    def test_returns_parsed_extract(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json=_build_response())

        with WikivoyageFetcher(
            lang="en",
            user_agent="test-ua/1.0",
            delay_s=0.0,
            client=_mock_client(handler),
        ) as fetcher:
            extract = fetcher.fetch_extract("Tokyo")

        assert extract is not None
        assert isinstance(extract, ArticleExtract)
        assert extract.requested_title == "Tokyo"
        assert extract.resolved_title == "Tokyo"
        assert extract.page_id == "12345"
        assert extract.language == "en"
        assert extract.extract == "Tokyo is the capital."
        assert "explaintext=1" in captured["url"]
        assert "redirects=1" in captured["url"]
        assert "formatversion=2" in captured["url"]
        assert captured["headers"]["user-agent"] == "test-ua/1.0"

    def test_returns_none_on_missing_page(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"query": {"pages": [{"title": "Nope", "missing": True}]}},
            )

        with WikivoyageFetcher(
            lang="en",
            user_agent="t/1",
            delay_s=0.0,
            client=_mock_client(handler),
        ) as fetcher:
            assert fetcher.fetch_extract("Nope") is None

    def test_returns_none_on_empty_extract(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_build_response(extract="   "))

        with WikivoyageFetcher(
            lang="en",
            user_agent="t/1",
            delay_s=0.0,
            client=_mock_client(handler),
        ) as fetcher:
            assert fetcher.fetch_extract("Tokyo") is None

    def test_resolves_via_redirect(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_build_response(title="New York City"),
            )

        with WikivoyageFetcher(
            lang="en",
            user_agent="t/1",
            delay_s=0.0,
            client=_mock_client(handler),
        ) as fetcher:
            extract = fetcher.fetch_extract("NYC")
        assert extract is not None
        assert extract.requested_title == "NYC"
        assert extract.resolved_title == "New York City"

    def test_unexpected_payload_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"oops": "bad"})

        with (
            WikivoyageFetcher(
                lang="en",
                user_agent="t/1",
                delay_s=0.0,
                client=_mock_client(handler),
            ) as fetcher,
            pytest.raises(WikivoyageFetchError),
        ):
            fetcher.fetch_extract("Tokyo")

    def test_http_error_propagates(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="overloaded")

        with (
            WikivoyageFetcher(
                lang="en",
                user_agent="t/1",
                delay_s=0.0,
                client=_mock_client(handler),
            ) as fetcher,
            pytest.raises(httpx.HTTPStatusError),
        ):
            fetcher.fetch_extract("Tokyo")


class TestThrottling:
    def test_zero_delay_does_not_sleep(self):
        # Smoke: many calls in a row finish quickly when delay is 0.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_build_response())

        with WikivoyageFetcher(
            lang="en",
            user_agent="t/1",
            delay_s=0.0,
            client=_mock_client(handler),
        ) as fetcher:
            for _ in range(5):
                assert fetcher.fetch_extract("Tokyo") is not None
