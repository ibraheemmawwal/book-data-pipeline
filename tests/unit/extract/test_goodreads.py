"""The Goodreads adapter's containment.

Goodreads is an unofficial contract used under an explicitly accepted risk. The
rules that make that acceptable — two gates, honest identity, no control
bypass, one request in flight, a circuit breaker, and caching only validated
results — are the entire justification for the integration, so they are tested
rather than asserted in a comment.
"""

from __future__ import annotations

import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pipeline.config import Settings
from pipeline.extract.base import Rejected
from pipeline.extract.goodreads import (
    MAX_DETAIL_ATTEMPTS,
    TRANSIENT_BACKOFF_SECONDS,
    GoodreadsExtractor,
    GoodreadsNotAcceptedError,
    GoodreadsResultCache,
    GoodreadsUnavailableError,
    map_payload,
)
from pipeline.extract.goodreads_parsers import (
    ISBN_SANITY_FLOOR,
    MIN_TITLE_SIMILARITY,
    is_plausible_isbn_match,
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
            # These exercise the breaker, so transient retries are off: with
            # them on a 503 never reaches the breaker, which is the point of
            # the retries and would make these tests measure nothing.
            "goodreads_transient_retries": 0,
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


class TestFinalEdgeCases:
    @respx.mock
    async def test_a_transport_failure_counts_towards_the_circuit(
        self, extractor: GoodreadsExtractor
    ) -> None:
        # A connection reset is a failure like any other; not counting it would
        # let a dead host be retried for every candidate in the run.
        respx.get(AUTOCOMPLETE).mock(side_effect=httpx.ConnectError("refused"))

        async with extractor.build_client() as client:
            for _ in range(2):
                with pytest.raises(GoodreadsUnavailableError, match="transport"):
                    await extractor.autocomplete(client, "dune")

        assert extractor.circuit_open

    @respx.mock
    async def test_an_ordinary_client_error_does_not_open_the_circuit(
        self, extractor: GoodreadsExtractor
    ) -> None:
        # A 404 is about this request, not about our access to the site.
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(404))

        async with extractor.build_client() as client:
            with pytest.raises(GoodreadsUnavailableError):
                await extractor.autocomplete(client, "dune")

        assert not extractor.circuit_open

    def test_the_instance_method_delegates_to_the_module_mapper(
        self, extractor: GoodreadsExtractor
    ) -> None:
        candidate = load("goodreads_autocomplete.json")[0]

        assert extractor.to_raw_book(candidate) == map_payload(candidate)

    def test_an_unparseable_aria_position_keeps_the_series_name(self) -> None:
        parsed = parse_aria_series("Book 1.2.3 in the Discworld series")

        assert parsed == ("Discworld", None)

    def test_an_empty_json_ld_script_is_skipped(self) -> None:
        html = """<script type="application/ld+json"></script>
        <script type="application/ld+json">{"@type": "Book", "name": "Dune"}</script>"""

        parsed = parse_json_ld(html)

        assert parsed is not None
        assert parsed["name"] == "Dune"


BOOK_SHOW = r".*/book/show/.*"
WORK_EDITIONS = r".*/work/editions/.*"

# Detail pages that yield something. Enrichment is a precondition now — a
# search card carries no publication year and one author — so a test whose
# subject is ranking or the ISBN floor has to let detail succeed, or it is
# testing the enrichment rule by accident.
BOOK_DETAIL_HTML = """<html><head>
<script type="application/ld+json">
{"@type": "Book", "name": "A Game of Thrones", "numberOfPages": 835,
 "description": "A tale of ice and fire.",
 "author": [{"@type": "Person", "name": "George R.R. Martin"}]}
</script></head><body></body></html>"""

EDITIONS_HTML = """<html><body><div data-testid="editionCell">
Published August 2003 by Bantam
ISBN 9780553588484
</div></body></html>"""


def serve_detail() -> None:
    """Mock both detail pages with content a parser can read."""
    respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(200, text=BOOK_DETAIL_HTML))
    respx.get(url__regex=WORK_EDITIONS).mock(return_value=httpx.Response(200, text=EDITIONS_HTML))


MARKUP = Path(__file__).parent.parent.parent / "fixtures"


class TestResolve:
    @respx.mock
    async def test_a_title_query_resolves_the_best_candidate(
        self, extractor: GoodreadsExtractor
    ) -> None:
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(200, json=load("goodreads_autocomplete.json"))
        )
        serve_detail()

        async with extractor.build_client() as client:
            book = await extractor.resolve(client, "A Game of Thrones by George R.R. Martin")

        assert book is not None
        assert book.source_id == "13496"

    @respx.mock
    async def test_an_empty_autocomplete_resolves_to_none(
        self, extractor: GoodreadsExtractor
    ) -> None:
        # A miss is the resolver's cue to fall back, not an error.
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(200, json=[]))

        async with extractor.build_client() as client:
            assert await extractor.resolve(client, "no such book") is None

    @respx.mock
    async def test_a_poor_match_is_rejected_by_the_threshold(
        self, extractor: GoodreadsExtractor
    ) -> None:
        # Returning an unrelated book would be worse than returning nothing.
        # Note the threshold is weak: normalised Levenshtein puts two unrelated
        # titles of similar length near 0.4, so this uses a clearly dissimilar
        # query rather than a merely wrong one.
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(200, json=load("goodreads_autocomplete.json"))
        )

        async with extractor.build_client() as client:
            assert await extractor.resolve(client, "Zzyzx") is None

    @respx.mock
    async def test_an_isbn_query_bypasses_the_ranking_threshold(
        self, extractor: GoodreadsExtractor
    ) -> None:
        # An ISBN is an exact identifier, so Goodreads' own answer beats string
        # similarity against a title we may have wrong. "Dune Messiah" scores
        # 0.67 against "Dune": under the 0.75 ranking threshold, so a title
        # query rejects it, and over the 0.3 sanity floor, so an ISBN query
        # keeps it.
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "bookId": "1",
                        "title": "Dune Messiah",
                        "bookTitleBare": "Dune Messiah",
                    }
                ],
            )
        )
        serve_detail()

        async with extractor.build_client() as client:
            by_title = await extractor.resolve(client, "Dune")
            by_isbn = await extractor.resolve(client, "Dune", isbn="9780441172719")

        assert by_title is None
        assert by_isbn is not None

    @respx.mock
    async def test_the_second_call_is_served_from_cache(
        self, extractor: GoodreadsExtractor
    ) -> None:
        route = respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(200, json=load("goodreads_autocomplete.json"))
        )
        serve_detail()

        async with extractor.build_client() as client:
            await extractor.resolve(client, "A Game of Thrones")
            await extractor.resolve(client, "A Game of Thrones")

        assert route.call_count == 1

    @respx.mock
    async def test_a_miss_is_never_cached(self, extractor: GoodreadsExtractor) -> None:
        # Caching an empty answer would turn a transient blip into a run-long
        # hole in the catalogue.
        route = respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(200, json=[]))

        async with extractor.build_client() as client:
            await extractor.resolve(client, "nothing")
            await extractor.resolve(client, "nothing")

        assert route.call_count == 2


class TestEnrichment:
    @respx.mock
    async def test_detail_pages_add_series_and_edition_facts(
        self, extractor: GoodreadsExtractor
    ) -> None:
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(200, json=load("goodreads_autocomplete.json"))
        )
        respx.get(url__regex=BOOK_SHOW).mock(
            return_value=httpx.Response(
                200, text=(MARKUP / "goodreads_book_detail.html").read_text()
            )
        )
        respx.get(url__regex=WORK_EDITIONS).mock(
            return_value=httpx.Response(
                200, text=(MARKUP / "goodreads_work_editions.html").read_text()
            )
        )

        async with extractor.build_client() as client:
            book = await extractor.resolve(client, "A Game of Thrones")

        assert book is not None
        assert book.isbns == ["9780553381689"]
        assert book.publisher == "Bantam Books"
        assert book.series[0].name == "A Song of Ice and Fire"

    @respx.mock
    async def test_one_surviving_page_still_enriches(self, extractor: GoodreadsExtractor) -> None:
        # gather(return_exceptions=True): a partial observation beats none.
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(200, json=load("goodreads_autocomplete.json"))
        )
        respx.get(url__regex=BOOK_SHOW).mock(
            return_value=httpx.Response(
                200, text=(MARKUP / "goodreads_book_detail.html").read_text()
            )
        )
        respx.get(url__regex=WORK_EDITIONS).mock(side_effect=httpx.ConnectError("down"))

        async with extractor.build_client() as client:
            book = await extractor.resolve(client, "A Game of Thrones")

        assert book is not None
        assert book.series[0].name == "A Song of Ice and Fire"
        assert book.isbns == []

    @respx.mock
    async def test_both_pages_failing_yields_no_record(self, extractor: GoodreadsExtractor) -> None:
        """Enrichment is a precondition, not a bonus.

        A search card is not a record of a book: Goodreads publishes no
        publication year outside the editions page, and the card names a single
        author where the work may have three. Keeping one anyway is how 480
        stored observations came to have no years between them, each looking
        like a successful resolution.
        """
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(200, json=load("goodreads_autocomplete.json"))
        )
        respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(404))
        respx.get(url__regex=WORK_EDITIONS).mock(return_value=httpx.Response(404))

        async with extractor.build_client() as client:
            book = await extractor.resolve(client, "A Game of Thrones")

        assert book is None

    @respx.mock
    async def test_detail_is_fetched_for_the_winner_only(
        self, extractor: GoodreadsExtractor
    ) -> None:
        # Fanning detail across every autocomplete result would multiply our
        # traffic against an unofficial source for no gain. When the winner's
        # pages answer, nothing below it is touched.
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(200, json=load("goodreads_autocomplete.json"))
        )
        book_route = respx.get(url__regex=BOOK_SHOW).mock(
            return_value=httpx.Response(200, text=BOOK_DETAIL_HTML)
        )
        respx.get(url__regex=WORK_EDITIONS).mock(
            return_value=httpx.Response(200, text=EDITIONS_HTML)
        )

        async with extractor.build_client() as client:
            await extractor.resolve(client, "A Game of Thrones")

        assert book_route.call_count == 1

    @respx.mock
    async def test_a_winner_with_no_detail_falls_through_to_the_next(
        self, extractor: GoodreadsExtractor
    ) -> None:
        """The fall-through the loop always documented and never performed.

        It returned the un-enriched card instead, so candidates two and three
        were unreachable and a failed detail fetch looked like a resolution.
        """
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(200, json=load("goodreads_autocomplete.json"))
        )
        book_route = respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(404))
        respx.get(url__regex=WORK_EDITIONS).mock(return_value=httpx.Response(404))

        async with extractor.build_client() as client:
            await extractor.resolve(client, "A Game of Thrones")

        assert book_route.call_count > 1, "gave up on the first candidate"
        assert book_route.call_count <= MAX_DETAIL_ATTEMPTS, "unbounded fan-out"


class TestEnrichmentReplay:
    """Stored enrichment has to survive a recompute.

    The load layer rebuilds canonical fields by replaying stored payloads
    through the source mapper. Anything the detail pages contributed that the
    mapper cannot read back is silently lost on the next ingest — a confirmed
    series would quietly downgrade to an inferred one.
    """

    def test_a_stored_detail_block_rebuilds_the_series(self) -> None:
        book = map_payload(
            {
                "bookId": "13496",
                "title": "A Game of Thrones",
                "_detail": {
                    "series_label": "Book 1 in the A Song of Ice and Fire series",
                    "series_id": "45175",
                },
            }
        )

        assert isinstance(book, RawBook)
        assert book.series[0].name == "A Song of Ice and Fire"
        assert book.series[0].source_series_id == "45175"
        assert book.series[0].confirmed is True

    def test_a_detail_block_without_an_id_stays_unconfirmed(self) -> None:
        book = map_payload(
            {
                "bookId": "1",
                "title": "X",
                "_detail": {"series_label": "Book 2 in the Discworld series"},
            }
        )

        assert isinstance(book, RawBook)
        assert book.series[0].confirmed is False

    def test_a_stored_edition_block_rebuilds_the_isbn(self) -> None:
        book = map_payload(
            {
                "bookId": "1",
                "title": "X",
                "_edition": {
                    "isbn13": "9780553381689",
                    "published": "August 4, 1997",
                    "publisher": "Bantam Books",
                },
            }
        )

        assert isinstance(book, RawBook)
        assert book.isbns == ["9780553381689"]
        assert book.published == "August 4, 1997"
        assert book.publisher == "Bantam Books"

    def test_a_payload_with_no_enrichment_is_unaffected(self) -> None:
        book = map_payload(load("goodreads_autocomplete.json")[0])

        assert isinstance(book, RawBook)
        assert book.series[0].confirmed is False
        assert book.isbns == []


class TestIsbnSanityFloor:
    """A loose guard on ISBN lookups.

    ISBN queries skip ranking because an exact identifier is better evidence
    than string similarity against a title we may have wrong. But providers can
    disagree about which book an ISBN denotes, and when the answer shares
    almost nothing with what was asked for, one of them is wrong — falling back
    to a documented source beats guessing which.
    """

    @respx.mock
    async def test_a_grossly_different_title_is_discarded(
        self, extractor: GoodreadsExtractor
    ) -> None:
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "bookId": "1",
                        "title": "A Completely Different Book",
                        "bookTitleBare": "A Completely Different Book",
                    }
                ],
            )
        )

        async with extractor.build_client() as client:
            assert await extractor.resolve(client, "Dune", isbn="9780441172719") is None

    @respx.mock
    async def test_a_matching_title_is_kept(self, extractor: GoodreadsExtractor) -> None:
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(
                200,
                json=[{"bookId": "1", "title": "Dune", "bookTitleBare": "Dune"}],
            )
        )
        serve_detail()

        async with extractor.build_client() as client:
            book = await extractor.resolve(client, "Dune", isbn="9780441172719")

        assert book is not None

    @respx.mock
    async def test_a_fuller_subtitle_is_still_accepted(self, extractor: GoodreadsExtractor) -> None:
        # Goodreads routinely holds a longer title than a dump edition does.
        # The floor is loose precisely so this is not mistaken for a mismatch.
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "bookId": "1",
                        "title": "A Brief History of Time",
                        "bookTitleBare": (
                            "A Brief History of Time: From the Big Bang to Black Holes"
                        ),
                    }
                ],
            )
        )
        serve_detail()

        async with extractor.build_client() as client:
            book = await extractor.resolve(client, "A Brief History of Time", isbn="9780553380163")

        assert book is not None

    def test_the_floor_is_far_below_the_ranking_threshold(self) -> None:
        # They answer different questions: one asks "is this the same book",
        # the other "which of these is the best match".
        assert ISBN_SANITY_FLOOR < MIN_TITLE_SIMILARITY

    @pytest.mark.parametrize(
        ("ours", "theirs", "plausible"),
        [
            ("Dune", "Dune", True),
            ("Dune", "Dune Messiah", True),
            ("A Brief History of Time", "A Brief History of Time: From the Big Bang", True),
            ("Social Psychology", "Social Psychology: Study Guide", True),
            ("Dune", "The Wind in the Willows", False),
            ("Herbs and Spices", "Quantum Chromodynamics", False),
        ],
    )
    def test_what_the_floor_does_and_does_not_catch(
        self, ours: str, theirs: str, plausible: bool
    ) -> None:
        # Documented rather than asserted in prose: a companion volume with a
        # similar title passes, because it is indistinguishable from a provider
        # holding a fuller subtitle.
        assert is_plausible_isbn_match(ours, theirs) is plausible

    def test_an_empty_title_is_not_grounds_for_rejection(self) -> None:
        # Nothing to compare is not evidence of a mismatch.
        assert is_plausible_isbn_match("", "Anything")
        assert is_plausible_isbn_match("Anything", "")


class TestTheAcceptHeaderPerEndpoint:
    """One header, every publication year in the catalogue.

    Goodreads' bot mitigation answers the same client differently depending on
    what it asks for. The shared client requests JSON, which the autocomplete
    and book endpoints answer with 200 — and which the work editions page
    answers with 404. Not a redirect, not a block: a plain 404 for a page that
    exists and returns 125KB of HTML the moment the header changes.

    Goodreads publishes no year anywhere except that page, so 480 stored
    records carried a workId, asked for its editions, were told it did not
    exist, and moved on without complaint. Nothing failed and nothing said so.
    """

    @respx.mock
    async def test_the_editions_page_is_asked_for_html(self, settings: Settings) -> None:
        respx.get("https://www.goodreads.com/book/auto_complete").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "bookId": "1",
                        "workId": "9",
                        "bookTitleBare": "Dune",
                        "author": {"name": "Frank Herbert"},
                    }
                ],
            )
        )
        respx.get("https://www.goodreads.com/book/show/1").mock(
            return_value=httpx.Response(200, text="<html></html>")
        )
        editions = respx.get("https://www.goodreads.com/work/editions/9").mock(
            return_value=httpx.Response(200, text="<html></html>")
        )

        extractor = GoodreadsExtractor(settings)
        async with extractor.build_client() as client:
            await extractor.resolve(client, "Dune")

        assert editions.called
        accept = editions.calls[0].request.headers.get("accept", "")
        assert "text/html" in accept, f"editions asked for {accept!r}; Goodreads 404s that"
        assert accept != "application/json"

    @respx.mock
    async def test_autocomplete_still_asks_for_json(self, settings: Settings) -> None:
        # The reverse mistake would be just as silent: the JSON endpoint
        # returns HTML when asked for anything else.
        auto = respx.get("https://www.goodreads.com/book/auto_complete").mock(
            return_value=httpx.Response(200, json=[])
        )

        extractor = GoodreadsExtractor(settings)
        async with extractor.build_client() as client:
            await extractor.autocomplete(client, "Dune")

        assert "application/json" in auto.calls[0].request.headers.get("accept", "")


class TestASoftBlock:
    """A 202 with an empty body.

    Observed live: /book/show/2657 served 849KB, and fifteen minutes later the
    same URL answered 202 with zero bytes, while /book/auto_complete kept
    returning results. Goodreads throttles its HTML pages and leaves the JSON
    endpoint alone.

    Every check in the client passed it: the status is not an error, the body
    holds no challenge markers, and there is nothing to reject. The parser then
    finds no fields and the record degrades to a thin one — a resolution the
    run reports as successful and that contains nothing.
    """

    @respx.mock
    async def test_an_empty_body_is_a_block_not_a_thin_page(
        self, extractor: GoodreadsExtractor
    ) -> None:
        respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(202, text=""))

        async with extractor.build_client() as client:
            with pytest.raises(GoodreadsUnavailableError, match="empty body"):
                await extractor._get(client, "/book/show/2657")

    @respx.mock
    async def test_one_empty_response_does_not_abandon_the_run(
        self, extractor: GoodreadsExtractor
    ) -> None:
        """One empty page proves a page is unusable, not that we are blocked.

        Tripping on sight abandoned a run of 36 books over a single odd one,
        which is inferring "blocked" from a sample of one.
        """
        respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(202, text="   "))

        async with extractor.build_client() as client:
            with pytest.raises(GoodreadsUnavailableError):
                await extractor._get(client, "/book/show/2657")

        assert not extractor.circuit_open

    @respx.mock
    async def test_a_run_of_empty_responses_does(self, extractor: GoodreadsExtractor) -> None:
        """Consecutive ones are the evidence.

        Measured live, a real block returned 202 and zero bytes for eight book
        pages out of eight, so the threshold is reached almost immediately when
        it genuinely is a block.
        """
        respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(202, text=""))

        async with extractor.build_client() as client:
            for _ in range(extractor._circuit.threshold):
                with pytest.raises(GoodreadsUnavailableError):
                    await extractor._get(client, "/book/show/2657")

        assert extractor.circuit_open

    @respx.mock
    async def test_a_success_in_between_resets_the_count(
        self, extractor: GoodreadsExtractor
    ) -> None:
        # Scattered empties across a healthy run must never accumulate into a
        # false positive.
        respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(202, text=""))
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(200, json=[]))

        async with extractor.build_client() as client:
            for _ in range(extractor._circuit.threshold - 1):
                with pytest.raises(GoodreadsUnavailableError):
                    await extractor._get(client, "/book/show/2657")
            await extractor.autocomplete(client, "anything")
            with pytest.raises(GoodreadsUnavailableError):
                await extractor._get(client, "/book/show/2657")

        assert not extractor.circuit_open

    @respx.mock
    async def test_a_small_but_real_answer_is_not_a_block(
        self, extractor: GoodreadsExtractor
    ) -> None:
        """Empty, not merely small.

        An autocomplete miss is two characters. A threshold generous enough to
        cover a book page would reject every one of them.
        """
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(200, json=[]))

        async with extractor.build_client() as client:
            assert await extractor.autocomplete(client, "nothing matches this") == []
        assert not extractor.circuit_open


class TestCompletingARecordFromABareId:
    """What an export record needs before it is worth loading.

    A title, its authors and a rating are what an export carries. The year, the
    ISBN, the page count and the series live on the book's own page, so an id
    has to become a request.
    """

    PAGE = """<html><head><script type="application/ld+json">
    {"@type": "Book", "name": "To Kill a Mockingbird", "isbn": "9780060935467",
     "numberOfPages": 323, "description": "A tale of Maycomb.",
     "author": [{"@type": "Person", "name": "Harper Lee"}]}
    </script></head><body>
    <a href="/work/editions/3275794">All editions</a>
    <script>{"publicationTime":1148367600000}</script>
    </body></html>"""

    THIN = """<html><head><script type="application/ld+json">
    {"@type": "Book", "name": "Nineteen Eighty-Four", "numberOfPages": 328}
    </script></head><body><a href="/work/editions/153313">All editions</a></body></html>"""

    EDITIONS = """<html><body><div data-testid="editionCell">
    Published June 1949 by Secker and Warburg
    ISBN 9780451524935
    </div></body></html>"""

    @staticmethod
    def _observation() -> RawBook:
        return RawBook(
            source=SourceName.GOODREADS,
            source_id="2657",
            title="To Kill a Mockingbird",
            raw_payload={"bookId": "2657", "title": "To Kill a Mockingbird"},
        )

    @respx.mock
    async def test_a_complete_page_needs_no_second_request(
        self, extractor: GoodreadsExtractor
    ) -> None:
        """The editions page is worth a request only when the first withheld
        something. Asking anyway would double the traffic against a source that
        has asked not to be crawled, for the sake of the minority."""
        respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(200, text=self.PAGE))
        editions = respx.get(url__regex=WORK_EDITIONS).mock(
            return_value=httpx.Response(200, text=self.EDITIONS)
        )

        async with extractor.build_client() as client:
            book = await extractor.enrich_by_id(client, self._observation())

        assert book is not None
        assert book.published == "2006"
        assert book.isbns == ["9780060935467"]
        assert book.page_count == 323
        assert not editions.called, "asked for editions it did not need"

    @respx.mock
    async def test_a_thin_page_falls_through_to_the_editions(
        self, extractor: GoodreadsExtractor
    ) -> None:
        respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(200, text=self.THIN))
        editions = respx.get(url__regex=WORK_EDITIONS).mock(
            return_value=httpx.Response(200, text=self.EDITIONS)
        )

        async with extractor.build_client() as client:
            book = await extractor.enrich_by_id(client, self._observation())

        assert editions.called
        assert book is not None
        assert book.isbns == ["9780451524935"]
        assert book.published == "June 1949"

    @respx.mock
    async def test_a_blocked_book_page_yields_nothing(self, extractor: GoodreadsExtractor) -> None:
        # Rather than a record that looks resolved and contains a title.
        respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(202, text=""))

        async with extractor.build_client() as client:
            assert await extractor.enrich_by_id(client, self._observation()) is None

    @respx.mock
    async def test_a_page_with_nothing_to_add_yields_nothing(
        self, extractor: GoodreadsExtractor
    ) -> None:
        respx.get(url__regex=BOOK_SHOW).mock(
            return_value=httpx.Response(200, text="<html><body>no data here at all</body></html>")
        )

        async with extractor.build_client() as client:
            assert await extractor.enrich_by_id(client, self._observation()) is None

    async def test_a_record_without_an_id_cannot_be_completed(
        self, extractor: GoodreadsExtractor
    ) -> None:
        observation = RawBook(source=SourceName.GOODREADS, source_id="x", title="T", raw_payload={})
        stripped = observation.model_copy(update={"source_id": ""})

        async with extractor.build_client() as client:
            assert await extractor.enrich_by_id(client, stripped) is None

    @respx.mock
    async def test_a_blocked_editions_page_keeps_what_the_book_page_gave(
        self, extractor: GoodreadsExtractor
    ) -> None:
        """The second request failing must not discard the first.

        Enrichment is cumulative: a page count and a description are worth
        keeping even when the ISBN never arrives.
        """
        respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(200, text=self.THIN))
        respx.get(url__regex=WORK_EDITIONS).mock(return_value=httpx.Response(503))

        async with extractor.build_client() as client:
            book = await extractor.enrich_by_id(client, self._observation())

        assert book is not None
        assert book.page_count == 328
        assert book.isbns == []

    @respx.mock
    async def test_the_book_page_wins_where_both_answer(
        self, extractor: GoodreadsExtractor
    ) -> None:
        # The editions list describes one printing; the book page describes the
        # book. Where they overlap, the book page is the better claim.
        page = self.PAGE.replace('"isbn": "9780060935467"', '"isbn": "9780061120084"')
        respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(200, text=page))
        respx.get(url__regex=WORK_EDITIONS).mock(
            return_value=httpx.Response(200, text=self.EDITIONS)
        )

        async with extractor.build_client() as client:
            book = await extractor.enrich_by_id(client, self._observation())

        assert book is not None
        assert book.isbns == ["9780061120084"]

    @respx.mock
    async def test_a_publisher_from_the_editions_page_is_kept(
        self, extractor: GoodreadsExtractor
    ) -> None:
        respx.get(url__regex=BOOK_SHOW).mock(return_value=httpx.Response(200, text=self.THIN))
        respx.get(url__regex=WORK_EDITIONS).mock(
            return_value=httpx.Response(200, text=self.EDITIONS)
        )

        async with extractor.build_client() as client:
            book = await extractor.enrich_by_id(client, self._observation())

        assert book is not None
        assert book.publisher is not None


@pytest.fixture
def retrying(accepted: Settings) -> GoodreadsExtractor:
    """An extractor with the real retry budget and no waiting."""

    async def no_wait(_: float) -> None:
        return None

    return GoodreadsExtractor(
        accepted.model_copy(update={"goodreads_transient_retries": 3}), sleep=no_wait
    )


class TestTransientFailuresAreNotRefusals:
    """A 503 is the source failing, not the source deciding about us.

    Measured live: Goodreads served 503 to one request in three while answering
    the other two with a complete page, and those 503s clustered — a sample of
    twelve contained a run of three against a breaker threshold of five. Read as
    a block, an ordinary wobble stopped the run and then kept every DAG away for
    ninety minutes from a source that was talking to us.
    """

    @respx.mock
    async def test_a_lone_503_is_retried_not_counted(self, retrying: GoodreadsExtractor) -> None:
        route = respx.get(AUTOCOMPLETE).mock(
            side_effect=[httpx.Response(503), httpx.Response(200, json=[])]
        )

        async with retrying.build_client() as client:
            await retrying.autocomplete(client, "dune")

        assert route.call_count == 2
        assert not retrying.circuit_open

    @respx.mock
    async def test_a_cluster_shorter_than_the_budget_survives(
        self, retrying: GoodreadsExtractor
    ) -> None:
        # The run of three actually observed, against three retries.
        respx.get(AUTOCOMPLETE).mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(503),
                httpx.Response(503),
                httpx.Response(200, json=[]),
            ]
        )

        async with retrying.build_client() as client:
            await retrying.autocomplete(client, "dune")

        assert not retrying.circuit_open

    @respx.mock
    async def test_exhausted_retries_open_the_circuit_but_not_as_a_refusal(
        self, retrying: GoodreadsExtractor
    ) -> None:
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(503))

        async with retrying.build_client() as client:
            for _ in range(2):
                with pytest.raises(GoodreadsUnavailableError):
                    await retrying.autocomplete(client, "dune")

        # The run stops — but the next one must be free to try, because
        # nothing here was a decision about us.
        assert retrying.circuit_open
        assert not retrying.refused

    @respx.mock
    async def test_a_transport_failure_is_transient_too(self, retrying: GoodreadsExtractor) -> None:
        respx.get(AUTOCOMPLETE).mock(
            side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json=[])]
        )

        async with retrying.build_client() as client:
            await retrying.autocomplete(client, "dune")

        assert not retrying.circuit_open


class TestRealRefusalsStillCount:
    @respx.mock
    async def test_access_denied_trips_immediately_as_a_refusal(
        self, retrying: GoodreadsExtractor
    ) -> None:
        route = respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(403))

        async with retrying.build_client() as client:
            with pytest.raises(GoodreadsUnavailableError):
                await retrying.autocomplete(client, "dune")

        # Never retried: repeating a block is the behaviour the containment
        # rules forbid, and it would turn one refusal into four.
        assert route.call_count == 1
        assert retrying.refused

    @respx.mock
    async def test_a_challenge_page_is_a_refusal(self, retrying: GoodreadsExtractor) -> None:
        respx.get(AUTOCOMPLETE).mock(
            return_value=httpx.Response(200, text="please complete the captcha")
        )

        async with retrying.build_client() as client:
            with pytest.raises(GoodreadsUnavailableError):
                await retrying.autocomplete(client, "dune")

        assert retrying.refused

    @respx.mock
    async def test_repeated_empty_bodies_are_a_refusal(self, retrying: GoodreadsExtractor) -> None:
        # The signature of the real block: 202 with zero bytes, eight pages
        # out of eight.
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(202, text=""))

        async with retrying.build_client() as client:
            for _ in range(2):
                with pytest.raises(GoodreadsUnavailableError):
                    await retrying.autocomplete(client, "dune")

        assert retrying.refused

    @respx.mock
    async def test_a_404_is_neither_retried_nor_counted(self, retrying: GoodreadsExtractor) -> None:
        # About the page, not about us. Retrying only reconfirms that the book
        # does not exist, and counting it would let a run of dead records in
        # the backlog look exactly like a block.
        route = respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(404))

        async with retrying.build_client() as client:
            for _ in range(4):
                with pytest.raises(GoodreadsUnavailableError):
                    await retrying.autocomplete(client, "dune")

        assert route.call_count == 4
        assert not retrying.circuit_open


class TestBackoffWaitsForReal:
    async def test_it_sleeps_when_no_clock_is_injected(self, accepted: Settings) -> None:
        # Every other test injects a no-op sleep, which would leave the real
        # wait — the thing that makes a retry polite rather than a second
        # hammer — never executed.
        extractor = GoodreadsExtractor(
            accepted.model_copy(update={"goodreads_transient_retries": 1})
        )
        started = time.monotonic()

        await extractor._backoff(1)

        assert time.monotonic() - started >= TRANSIENT_BACKOFF_SECONDS
