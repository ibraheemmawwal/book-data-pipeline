"""Open Library extractor.

Open Library asks that its API not be used for bulk harvesting, so this is a
bounded enrichment source: identified requests, one per second, an explicit
field list, and a hard record budget.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from pipeline.config import Settings
from pipeline.extract.base import ExtractionRequest, Rejected
from pipeline.extract.openlibrary import OpenLibraryExtractor
from pipeline.models.domain import RawBook, SourceName

from .conftest import load_fixture

SEARCH = "https://openlibrary.org/search.json"


async def collect(extractor: OpenLibraryExtractor, limit: int = 100) -> list[RawBook | Rejected]:
    return [item async for item in extractor.fetch(ExtractionRequest(max_records=limit))]


@pytest.fixture
def extractor(settings: Settings) -> OpenLibraryExtractor:
    async def no_wait(_: float) -> None:
        return None

    return OpenLibraryExtractor(settings, base_delay=0.0, sleep=no_wait)


class TestUsagePolicy:
    @respx.mock
    async def test_identifies_itself_with_a_contact_address(
        self, extractor: OpenLibraryExtractor
    ) -> None:
        route = respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json=load_fixture("openlibrary_empty.json"))
        )
        await collect(extractor)

        agent = route.calls[0].request.headers["user-agent"]
        assert "book-data-pipeline" in agent
        assert "owner@example.com" in agent

    @respx.mock
    async def test_requests_an_explicit_field_list(self, extractor: OpenLibraryExtractor) -> None:
        # Without `fields`, search returns a thin document and the enrichment
        # is pointless; with it, one request carries what we actually need.
        route = respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json=load_fixture("openlibrary_empty.json"))
        )
        await collect(extractor)

        fields = route.calls[0].request.url.params["fields"]
        for required in (
            "key",
            "title",
            "author_name",
            "author_key",
            "first_publish_year",
            "isbn",
            "language",
            "number_of_pages_median",
            "publisher",
            "subject",
        ):
            assert required in fields

    @respx.mock
    async def test_rate_limits_between_pages(self, settings: Settings) -> None:
        slept: list[float] = []

        async def record(delay: float) -> None:
            slept.append(delay)

        extractor = OpenLibraryExtractor(settings, base_delay=0.0, sleep=record, page_size=3)
        respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json=load_fixture("openlibrary_search.json"))
        )
        await collect(extractor, limit=9)

        # A three-record page size against a nine-record budget needs three
        # requests and therefore two enforced gaps. Each is a shade under a
        # second: time already spent on the request is subtracted from the wait.
        assert len(slept) == 2
        assert all(d == pytest.approx(1.0, abs=0.05) for d in slept)

    @respx.mock
    async def test_never_exceeds_the_configured_record_budget(
        self, extractor: OpenLibraryExtractor
    ) -> None:
        respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json=load_fixture("openlibrary_search.json"))
        )

        assert len(await collect(extractor, limit=2)) == 2


class TestMapping:
    @respx.mock
    async def test_maps_a_captured_document(self, extractor: OpenLibraryExtractor) -> None:
        respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json=load_fixture("openlibrary_search.json"))
        )

        books = [b for b in await collect(extractor, limit=3) if isinstance(b, RawBook)]
        dune = books[0]

        assert dune.source is SourceName.OPENLIBRARY
        assert dune.title
        assert dune.published == "1965"
        assert dune.authors[0].name == "Frank Herbert"
        assert len(dune.isbns) > 1

    @respx.mock
    async def test_source_id_is_the_work_key(self, extractor: OpenLibraryExtractor) -> None:
        respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json=load_fixture("openlibrary_search.json"))
        )

        books = [b for b in await collect(extractor, limit=1) if isinstance(b, RawBook)]

        # The work key is stable across editions; a title is not an identifier.
        assert books[0].source_id.startswith("/works/OL")

    @respx.mock
    async def test_author_keys_are_kept_for_cross_source_identity(
        self, extractor: OpenLibraryExtractor
    ) -> None:
        respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json=load_fixture("openlibrary_search.json"))
        )

        books = [b for b in await collect(extractor, limit=1) if isinstance(b, RawBook)]

        assert books[0].authors[0].source_author_id is not None
        assert books[0].authors[0].source_author_id.startswith("OL")

    @respx.mock
    async def test_author_names_and_keys_are_paired_positionally(
        self, extractor: OpenLibraryExtractor
    ) -> None:
        # author_name and author_key are parallel arrays. Zipping them wrongly
        # attributes books to the wrong person, which is invisible downstream.
        respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json=load_fixture("openlibrary_search.json"))
        )

        books = [b for b in await collect(extractor, limit=1) if isinstance(b, RawBook)]
        doc = load_fixture("openlibrary_search.json")["docs"][0]

        for author, name, key in zip(
            books[0].authors, doc["author_name"], doc["author_key"], strict=True
        ):
            assert author.name == name
            assert author.source_author_id == key

    @respx.mock
    async def test_mismatched_author_arrays_do_not_crash(
        self, extractor: OpenLibraryExtractor
    ) -> None:
        # Real documents sometimes carry more names than keys.
        doc = load_fixture("openlibrary_search.json")["docs"][0]
        doc = {**doc, "author_name": ["A", "B", "C"], "author_key": ["OL1A"]}
        respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json={"numFound": 1, "docs": [doc]})
        )

        books = [b for b in await collect(extractor, limit=1) if isinstance(b, RawBook)]

        assert [a.name for a in books[0].authors] == ["A", "B", "C"]
        assert [a.source_author_id for a in books[0].authors] == ["OL1A", None, None]

    @respx.mock
    async def test_every_edition_language_is_preserved(
        self, extractor: OpenLibraryExtractor
    ) -> None:
        # A work carries one language per edition. Picking the first tagged
        # Dorian Gray as Czech in a live run, so the whole list is kept and
        # transform decides what, if anything, it means.
        doc = load_fixture("openlibrary_search.json")["docs"][0]
        doc = {**doc, "language": ["eng", "cze", "fre"]}
        respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json={"numFound": 1, "docs": [doc]})
        )

        books = [b for b in await collect(extractor, limit=1) if isinstance(b, RawBook)]

        assert books[0].languages == ["eng", "cze", "fre"]

    @respx.mock
    async def test_carries_no_author_lifespan(self, extractor: OpenLibraryExtractor) -> None:
        # Search results have no birth or death year; only Gutendex supplies it.
        respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json=load_fixture("openlibrary_search.json"))
        )

        books = [b for b in await collect(extractor, limit=1) if isinstance(b, RawBook)]

        assert all(a.birth_year is None for a in books[0].authors)


class TestPagination:
    @respx.mock
    async def test_advances_the_page_parameter(self, settings: Settings) -> None:
        async def no_wait(_: float) -> None:
            return None

        extractor = OpenLibraryExtractor(settings, base_delay=0.0, sleep=no_wait, page_size=3)
        route = respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json=load_fixture("openlibrary_search.json"))
        )
        await collect(extractor, limit=6)

        assert [c.request.url.params["page"] for c in route.calls] == ["1", "2"]

    @respx.mock
    async def test_stops_on_an_empty_page(self, extractor: OpenLibraryExtractor) -> None:
        route = respx.get(SEARCH).mock(
            return_value=httpx.Response(200, json=load_fixture("openlibrary_empty.json"))
        )

        assert await collect(extractor, limit=50) == []
        assert route.call_count == 1


class TestPerItemIsolation:
    @respx.mock
    async def test_a_document_without_a_key_is_rejected(
        self, extractor: OpenLibraryExtractor
    ) -> None:
        payload = {
            "numFound": 2,
            "docs": [{"title": "No key here"}, {"key": "/works/OL1W", "title": "Fine"}],
        }
        respx.get(SEARCH).mock(return_value=httpx.Response(200, json=payload))

        items = await collect(extractor, limit=2)

        assert len([i for i in items if isinstance(i, RawBook)]) == 1
        assert len([i for i in items if isinstance(i, Rejected)]) == 1
