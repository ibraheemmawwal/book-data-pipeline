"""Gutendex extractor.

The primary bulk source. It supplies no ISBN, publisher, page count or
publication year, so the fields it *does* carry densely — author lifespan,
subjects and download counts — have to survive mapping intact.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from pipeline.config import Settings
from pipeline.extract.base import ExtractionRequest, Rejected, SourceUnavailableError
from pipeline.extract.gutendex import GutendexExtractor
from pipeline.models.domain import RawBook, SourceName

from .conftest import load_fixture

BOOKS = "https://gutendex.com/books"
PAGE2 = "https://gutendex.com/books/"


async def collect(extractor: GutendexExtractor, limit: int = 100) -> list[RawBook | Rejected]:
    return [item async for item in extractor.fetch(ExtractionRequest(max_records=limit))]


@pytest.fixture
def extractor(settings: Settings) -> GutendexExtractor:
    return GutendexExtractor(settings, base_delay=0.0)


class TestMapping:
    @respx.mock
    async def test_maps_a_captured_record(self, extractor: GutendexExtractor) -> None:
        respx.get(BOOKS).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_page1.json"))
        )
        respx.get(PAGE2).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_page2.json"))
        )

        books = [b for b in await collect(extractor) if isinstance(b, RawBook)]
        moby = next(b for b in books if b.source_id == "2701")

        assert moby.source is SourceName.GUTENDEX
        assert moby.title == "Moby Dick; Or, The Whale"
        assert moby.authors[0].name == "Melville, Herman"
        assert moby.authors[0].birth_year == 1819
        assert moby.authors[0].death_year == 1891
        assert "Adventure stories" in moby.subjects
        assert moby.download_count is not None
        assert moby.download_count > 0

    @respx.mock
    async def test_summary_becomes_the_description(self, extractor: GutendexExtractor) -> None:
        # Gutendex is the only source that gives us description text for the
        # weight-C component of the search vector.
        respx.get(BOOKS).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_page1.json"))
        )
        respx.get(PAGE2).mock(return_value=httpx.Response(200, json={"results": [], "next": None}))

        books = [b for b in await collect(extractor) if isinstance(b, RawBook)]

        assert books[0].description is not None
        assert len(books[0].description) > 50

    @respx.mock
    async def test_first_non_blank_summary_becomes_description(
        self, extractor: GutendexExtractor
    ) -> None:
        payload = {
            "results": [
                {
                    "id": 1,
                    "title": "A book",
                    "summaries": ["", "  ", "Useful summary"],
                }
            ],
            "next": None,
        }
        respx.get(BOOKS).mock(return_value=httpx.Response(200, json=payload))

        books = [book for book in await collect(extractor) if isinstance(book, RawBook)]

        assert books[0].description == "Useful summary"

    @respx.mock
    async def test_languages_are_left_untranslated_for_transform(
        self, extractor: GutendexExtractor
    ) -> None:
        # Gutendex speaks ISO 639-1 ("en"); mapping to 639-3 is transform's job
        # and doing it here would put a lookup table in the I/O layer.
        respx.get(BOOKS).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_page1.json"))
        )
        respx.get(PAGE2).mock(return_value=httpx.Response(200, json={"results": [], "next": None}))

        books = [b for b in await collect(extractor) if isinstance(b, RawBook)]

        assert books[0].languages == ["en"]

    @respx.mock
    async def test_cover_url_comes_from_the_jpeg_format(self, extractor: GutendexExtractor) -> None:
        respx.get(BOOKS).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_page1.json"))
        )
        respx.get(PAGE2).mock(return_value=httpx.Response(200, json={"results": [], "next": None}))

        books = [b for b in await collect(extractor) if isinstance(b, RawBook)]

        assert books[0].cover_url is not None
        assert ".jpg" in books[0].cover_url or "cover" in books[0].cover_url

    @respx.mock
    async def test_carries_no_isbn_publisher_year_or_page_count(
        self, extractor: GutendexExtractor
    ) -> None:
        # Documented reality, asserted so a future refactor cannot quietly
        # invent values the source never supplied.
        respx.get(BOOKS).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_page1.json"))
        )
        respx.get(PAGE2).mock(return_value=httpx.Response(200, json={"results": [], "next": None}))

        books = [b for b in await collect(extractor) if isinstance(b, RawBook)]

        assert all(b.isbns == [] for b in books)
        assert all(b.publisher is None for b in books)
        assert all(b.published is None for b in books)
        assert all(b.page_count is None for b in books)

    @respx.mock
    async def test_raw_payload_is_the_untouched_source_record(
        self, extractor: GutendexExtractor
    ) -> None:
        page = load_fixture("gutendex_page1.json")
        respx.get(BOOKS).mock(return_value=httpx.Response(200, json=page))
        respx.get(PAGE2).mock(return_value=httpx.Response(200, json={"results": [], "next": None}))

        books = [b for b in await collect(extractor) if isinstance(b, RawBook)]

        assert books[0].raw_payload == page["results"][0]


class TestPagination:
    @respx.mock
    async def test_follows_the_next_link_rather_than_building_page_numbers(
        self, extractor: GutendexExtractor
    ) -> None:
        first = respx.get(BOOKS).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_page1.json"))
        )
        second = respx.get(PAGE2, params={"page": "2"}).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_page2.json"))
        )

        books = await collect(extractor)

        assert first.called
        assert second.called
        assert len(books) == 5

    @respx.mock
    async def test_stops_at_the_record_limit_mid_page(self, extractor: GutendexExtractor) -> None:
        respx.get(BOOKS).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_page1.json"))
        )
        page2 = respx.get(PAGE2).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_page2.json"))
        )

        books = await collect(extractor, limit=2)

        assert len(books) == 2
        # The limit is a budget, not a filter: page 2 is never requested.
        assert not page2.called

    @respx.mock
    async def test_terminates_on_a_null_next_link(self, extractor: GutendexExtractor) -> None:
        respx.get(BOOKS).mock(
            return_value=httpx.Response(
                200, json={"count": 1, "next": None, "previous": None, "results": []}
            )
        )

        assert await collect(extractor) == []


class TestPerItemIsolation:
    @respx.mock
    async def test_a_bad_record_is_rejected_without_losing_the_page(
        self, extractor: GutendexExtractor
    ) -> None:
        respx.get(BOOKS).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_malformed_items.json"))
        )

        items = await collect(extractor)
        books = [i for i in items if isinstance(i, RawBook)]
        rejects = [i for i in items if isinstance(i, Rejected)]

        assert len(books) == 1
        assert len(rejects) == 2
        assert all(r.source is SourceName.GUTENDEX for r in rejects)

    @respx.mock
    async def test_a_rejection_keeps_the_payload_that_caused_it(
        self, extractor: GutendexExtractor
    ) -> None:
        respx.get(BOOKS).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_malformed_items.json"))
        )

        rejects = [i for i in await collect(extractor) if isinstance(i, Rejected)]

        assert all(r.raw_payload for r in rejects)
        assert all(r.detail for r in rejects)

    @respx.mock
    async def test_a_record_with_no_id_is_rejected_with_no_source_id(
        self, extractor: GutendexExtractor
    ) -> None:
        respx.get(BOOKS).mock(
            return_value=httpx.Response(200, json=load_fixture("gutendex_malformed_items.json"))
        )

        rejects = [i for i in await collect(extractor) if isinstance(i, Rejected)]

        assert any(r.source_id is None for r in rejects)

    @respx.mock
    async def test_structurally_invalid_item_is_rejected_without_losing_the_page(
        self, extractor: GutendexExtractor
    ) -> None:
        payload = {
            "results": [
                {"id": 1, "title": "Bad", "authors": "not-an-array"},
                {"id": 2, "title": "Good"},
            ],
            "next": None,
        }
        respx.get(BOOKS).mock(return_value=httpx.Response(200, json=payload))

        items = await collect(extractor)

        assert len([item for item in items if isinstance(item, Rejected)]) == 1
        assert len([item for item in items if isinstance(item, RawBook)]) == 1

    @respx.mock
    async def test_non_object_item_is_preserved_as_a_rejection(
        self, extractor: GutendexExtractor
    ) -> None:
        respx.get(BOOKS).mock(
            return_value=httpx.Response(200, json={"results": ["bad"], "next": None})
        )

        items = await collect(extractor)

        assert isinstance(items[0], Rejected)
        assert items[0].raw_payload == "bad"


class TestFailure:
    @respx.mock
    async def test_a_terminal_failure_is_typed(self, extractor: GutendexExtractor) -> None:
        respx.get(BOOKS).mock(return_value=httpx.Response(500))

        with pytest.raises(SourceUnavailableError) as caught:
            await collect(extractor)

        assert caught.value.source == SourceName.GUTENDEX.value

    @respx.mock
    async def test_a_non_json_body_is_a_source_failure_not_a_crash(
        self, extractor: GutendexExtractor
    ) -> None:
        respx.get(BOOKS).mock(return_value=httpx.Response(200, text="<html>502</html>"))

        with pytest.raises(SourceUnavailableError):
            await collect(extractor)


class TestItSearchesForTheCandidate:
    """Gutendex is a resolver here, not a catalogue reader.

    Without a search term the endpoint returns page one of its default
    listing — the same book every time. That is not a thin result, it is the
    wrong book: every candidate "resolved" to Gutenberg id 2701.

    Six hundred resolved attempts in live runs left exactly one row in
    book_sources. That ratio is the signature to watch for: a source reporting
    resolutions it is not depositing.
    """

    @respx.mock
    async def test_the_candidate_query_is_sent(self, settings: Settings) -> None:
        route = respx.get("https://gutendex.com/books").mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )

        async for _ in GutendexExtractor(settings).fetch(
            ExtractionRequest(max_records=1, query="Pride and Prejudice Austen")
        ):
            pass

        assert route.called
        sent = route.calls[0].request.url.params
        assert sent.get("search") == "Pride and Prejudice Austen", (
            "no search term: every candidate resolves to the same book"
        )

    @respx.mock
    async def test_without_a_query_it_does_not_invent_one(self, settings: Settings) -> None:
        # Bulk discovery legitimately pages the default listing; only the
        # resolver path supplies a query.
        route = respx.get("https://gutendex.com/books").mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )

        async for _ in GutendexExtractor(settings).fetch(ExtractionRequest(max_records=1)):
            pass

        assert "search" not in route.calls[0].request.url.params

    @respx.mock
    async def test_the_next_link_is_followed_as_given(self, settings: Settings) -> None:
        """The search term must not be re-appended to a paged URL.

        Gutendex bakes the query into ``next``; sending params alongside it
        would either duplicate them or contradict the page it points at.
        """
        respx.get("https://gutendex.com/books").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "next": "https://gutendex.com/books?page=2&search=dune",
                        "results": [],
                    },
                ),
                httpx.Response(200, json={"next": None, "results": []}),
            ]
        )

        async for _ in GutendexExtractor(settings).fetch(
            ExtractionRequest(max_records=50, query="dune")
        ):
            pass
