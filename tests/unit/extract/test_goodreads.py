"""The Goodreads adapter's containment.

Goodreads is an unofficial contract used under an explicitly accepted risk. The
rules that make that acceptable — two gates, honest identity, no control
bypass, one request in flight, a circuit breaker, and caching only validated
results — are the entire justification for the integration, so they are tested
rather than asserted in a comment.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pipeline.config import Settings
from pipeline.extract.base import Rejected
from pipeline.extract.goodreads import (
    GoodreadsExtractor,
    GoodreadsNotAcceptedError,
    GoodreadsResultCache,
    GoodreadsUnavailableError,
    map_payload,
    parse_aria_series,
    parse_json_ld,
    parse_series_id,
)
from pipeline.models.domain import RawBook, SourceName

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
AUTOCOMPLETE = "https://www.goodreads.com/book/auto_complete"


def load(name: str) -> Any:
    with (FIXTURES / name).open() as handle:
        return json.load(handle)


@pytest.fixture
def accepted(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "goodreads_enabled": True,
            "goodreads_unofficial_source_accepted": True,
            "goodreads_circuit_failure_threshold": 2,
        }
    )


@pytest.fixture
def extractor(accepted: Settings) -> GoodreadsExtractor:
    async def no_wait(_: float) -> None:
        return None

    return GoodreadsExtractor(accepted, sleep=no_wait)


class TestGates:
    def test_disabled_by_default(self, settings: Settings) -> None:
        with pytest.raises(GoodreadsNotAcceptedError, match="disabled"):
            GoodreadsExtractor(settings).ensure_accepted()

    def test_enabling_alone_is_not_enough(self, settings: Settings) -> None:
        # Reading an unofficial contract takes a second, deliberate act.
        enabled = settings.model_copy(update={"goodreads_enabled": True})

        with pytest.raises(GoodreadsNotAcceptedError, match="risk has not been accepted"):
            GoodreadsExtractor(enabled).ensure_accepted()

    def test_both_gates_allow_it(self, extractor: GoodreadsExtractor) -> None:
        extractor.ensure_accepted()


class TestHonestIdentity:
    @respx.mock
    async def test_identifies_itself_rather_than_imitating_a_browser(
        self, extractor: GoodreadsExtractor
    ) -> None:
        route = respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(200, json=[]))
        async with extractor.build_client() as client:
            await extractor.autocomplete(client, "dune")

        agent = route.calls[0].request.headers["user-agent"]
        assert "book-data-pipeline" in agent
        assert "Mozilla" not in agent

    @respx.mock
    async def test_format_json_is_always_sent(self, extractor: GoodreadsExtractor) -> None:
        # Without it the route returns HTML and the parse fails confusingly.
        route = respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(200, json=[]))
        async with extractor.build_client() as client:
            await extractor.autocomplete(client, "dune")

        assert route.calls[0].request.url.params["format"] == "json"


class TestNoControlBypass:
    @respx.mock
    @pytest.mark.parametrize("status", [401, 403, 429])
    async def test_a_block_opens_the_circuit_immediately(
        self, extractor: GoodreadsExtractor, status: int
    ) -> None:
        # A block is an answer, not an error to retry around.
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(status))

        async with extractor.build_client() as client:
            with pytest.raises(GoodreadsUnavailableError):
                await extractor.autocomplete(client, "dune")

        assert extractor.circuit_open

    @respx.mock
    async def test_a_block_is_never_retried(self, extractor: GoodreadsExtractor) -> None:
        route = respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(403))

        async with extractor.build_client() as client:
            with pytest.raises(GoodreadsUnavailableError):
                await extractor.autocomplete(client, "dune")

        assert route.call_count == 1

    @respx.mock
    async def test_a_challenge_page_opens_the_circuit(self, extractor: GoodreadsExtractor) -> None:
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(200, text="<html>Please complete the CAPTCHA</html>")
        )

        async with extractor.build_client() as client:
            with pytest.raises(GoodreadsUnavailableError):
                await extractor.autocomplete(client, "dune")

        assert extractor.circuit_open


class TestCircuitBreaker:
    @respx.mock
    async def test_repeated_server_errors_open_it(self, extractor: GoodreadsExtractor) -> None:
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(503))

        async with extractor.build_client() as client:
            for _ in range(2):
                with pytest.raises(GoodreadsUnavailableError):
                    await extractor.autocomplete(client, "dune")

        assert extractor.circuit_open

    @respx.mock
    async def test_an_open_circuit_stops_further_requests(
        self, extractor: GoodreadsExtractor
    ) -> None:
        # One upstream outage must not become thousands of failing calls.
        route = respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(503))

        async with extractor.build_client() as client:
            for _ in range(5):
                with pytest.raises(GoodreadsUnavailableError):
                    await extractor.autocomplete(client, "dune")

        assert route.call_count == 2

    @respx.mock
    async def test_a_success_resets_the_failure_count(self, extractor: GoodreadsExtractor) -> None:
        respx.get(AUTOCOMPLETE).mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(200, json=[]),
                httpx.Response(503),
            ]
        )
        async with extractor.build_client() as client:
            with pytest.raises(GoodreadsUnavailableError):
                await extractor.autocomplete(client, "a")
            await extractor.autocomplete(client, "b")
            with pytest.raises(GoodreadsUnavailableError):
                await extractor.autocomplete(client, "c")

        assert not extractor.circuit_open


class TestAutocomplete:
    @respx.mock
    async def test_parses_a_captured_response(self, extractor: GoodreadsExtractor) -> None:
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(200, json=load("goodreads_autocomplete.json"))
        )
        async with extractor.build_client() as client:
            results = await extractor.autocomplete(client, "a game of thrones")

        assert len(results) == 3
        assert results[0]["bookId"] == "13496"

    @respx.mock
    async def test_a_non_json_body_is_a_source_failure(self, extractor: GoodreadsExtractor) -> None:
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(200, text="<html></html>"))

        async with extractor.build_client() as client:
            with pytest.raises(GoodreadsUnavailableError, match="not JSON"):
                await extractor.autocomplete(client, "dune")

    @respx.mock
    async def test_an_empty_result_is_not_an_error(self, extractor: GoodreadsExtractor) -> None:
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(200, json=[]))

        async with extractor.build_client() as client:
            assert await extractor.autocomplete(client, "zzz") == []


class TestMapping:
    def test_maps_a_captured_candidate(self) -> None:
        book = map_payload(load("goodreads_autocomplete.json")[0])

        assert isinstance(book, RawBook)
        assert book.source is SourceName.GOODREADS
        assert book.source_id == "13496"
        assert book.title == "A Game of Thrones"
        assert book.goodreads_average_rating == Decimal("4.45")
        assert book.page_count == 835

    def test_the_series_comes_out_of_the_dirty_title(self) -> None:
        book = map_payload(load("goodreads_autocomplete.json")[0])

        assert isinstance(book, RawBook)
        assert book.series[0].name == "A Song of Ice and Fire"
        assert book.series[0].position == "1"
        # Inferred from title text, not confirmed by a /series/ link.
        assert book.series[0].confirmed is False

    def test_the_cover_thumbnail_segment_is_stripped(self) -> None:
        book = map_payload(load("goodreads_autocomplete.json")[0])

        assert isinstance(book, RawBook)
        assert book.cover_url is not None
        assert "_SY75_" not in book.cover_url

    def test_a_non_object_payload_is_rejected(self) -> None:
        rejected = map_payload("not an object")

        assert isinstance(rejected, Rejected)
        assert rejected.source is SourceName.GOODREADS

    def test_a_candidate_without_a_book_id_is_rejected(self) -> None:
        assert isinstance(map_payload({"title": "No id"}), Rejected)

    def test_an_absent_rating_is_none_not_zero(self) -> None:
        # Zero is a real rating; absent is not, and conflating them would drag
        # every unrated book's average down.
        book = map_payload({"bookId": "1", "title": "X"})

        assert isinstance(book, RawBook)
        assert book.goodreads_average_rating is None


class TestResultCache:
    def test_a_validated_result_is_returned(self) -> None:
        cache = GoodreadsResultCache(title_ttl=3600, isbn_ttl=86400)
        book = map_payload(load("goodreads_autocomplete.json")[0])
        assert isinstance(book, RawBook)

        cache.put("k", book, is_isbn=False, now=0.0)

        assert cache.get("k", now=10.0) is book

    def test_an_expired_entry_is_dropped(self) -> None:
        cache = GoodreadsResultCache(title_ttl=60, isbn_ttl=60)
        book = map_payload(load("goodreads_autocomplete.json")[0])
        assert isinstance(book, RawBook)
        cache.put("k", book, is_isbn=False, now=0.0)

        assert cache.get("k", now=61.0) is None

    def test_isbn_lookups_live_longer_than_title_lookups(self) -> None:
        # An ISBN is an exact identifier; its answer does not drift.
        cache = GoodreadsResultCache(title_ttl=60, isbn_ttl=600)
        book = map_payload(load("goodreads_autocomplete.json")[0])
        assert isinstance(book, RawBook)
        cache.put("title", book, is_isbn=False, now=0.0)
        cache.put("isbn", book, is_isbn=True, now=0.0)

        assert cache.get("title", now=100.0) is None
        assert cache.get("isbn", now=100.0) is book

    def test_a_zero_ttl_disables_caching(self) -> None:
        cache = GoodreadsResultCache(title_ttl=0, isbn_ttl=0)
        book = map_payload(load("goodreads_autocomplete.json")[0])
        assert isinstance(book, RawBook)
        cache.put("k", book, is_isbn=False, now=0.0)

        assert cache.get("k", now=0.0) is None

    def test_a_miss_is_none(self) -> None:
        assert GoodreadsResultCache(60, 60).get("absent") is None


class TestSeriesIdConfirmation:
    def test_a_matching_slug_confirms_the_id(self) -> None:
        assert (
            parse_series_id("/series/45175-a-song-of-ice-and-fire", "A Song of Ice and Fire")
            == "45175"
        )

    def test_a_mismatched_slug_is_refused(self) -> None:
        # An id from an unrelated link would attach the book to the wrong
        # series permanently, and nothing downstream could detect it.
        assert parse_series_id("/series/999-the-lord-of-the-rings", "Discworld") is None

    def test_an_id_without_a_slug_is_accepted(self) -> None:
        assert parse_series_id("/series/45175", "Anything") == "45175"

    @pytest.mark.parametrize("href", [None, "", "/book/show/1", "not-a-url"])
    def test_a_non_series_href_yields_nothing(self, href: str | None) -> None:
        assert parse_series_id(href, "Discworld") is None


class TestAriaSeries:
    @pytest.mark.parametrize(
        ("label", "name", "position"),
        [
            ("Book 1 in the Discworld series", "Discworld", "1"),
            ("Book 2.5 in the Foundation series", "Foundation", "2.5"),
            ("Book 0.5 in the A Song of Ice and Fire series", "A Song of Ice and Fire", "0.5"),
        ],
    )
    def test_reads_name_and_decimal_position(self, label: str, name: str, position: str) -> None:
        parsed = parse_aria_series(label)

        assert parsed == (name, Decimal(position))

    @pytest.mark.parametrize("label", [None, "", "Some other label"])
    def test_an_unrecognised_label_yields_nothing(self, label: str | None) -> None:
        assert parse_aria_series(label) is None


class TestJsonLd:
    def test_extracts_a_book_block(self) -> None:
        html = """<html><head>
        <script type="application/ld+json">
        {"@type": "Book", "name": "Dune", "numberOfPages": 412}
        </script></head><body></body></html>"""

        parsed = parse_json_ld(html)

        assert parsed is not None
        assert parsed["name"] == "Dune"

    def test_ignores_non_book_blocks(self) -> None:
        html = """<script type="application/ld+json">
        {"@type": "WebSite", "name": "Goodreads"}</script>"""

        assert parse_json_ld(html) is None

    def test_malformed_json_does_not_raise(self) -> None:
        # An undocumented contract must fail closed, not explode.
        assert parse_json_ld('<script type="application/ld+json">{not json</script>') is None

    def test_a_page_without_json_ld_yields_none(self) -> None:
        assert parse_json_ld("<html><body>nothing here</body></html>") is None
