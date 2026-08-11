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
from pipeline.extract import openlibrary
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
    async def test_rate_limit_applies_to_retry_attempts(self, settings: Settings) -> None:
        slept: list[float] = []

        async def record(delay: float) -> None:
            slept.append(delay)

        extractor = OpenLibraryExtractor(settings, base_delay=0.0, sleep=record)
        route = respx.get(SEARCH).mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(200, json=load_fixture("openlibrary_empty.json")),
            ]
        )

        await collect(extractor)

        assert route.call_count == 2
        assert any(delay == pytest.approx(1.0, abs=0.05) for delay in slept)

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

    @respx.mock
    async def test_non_array_subject_is_rejected_without_losing_the_page(
        self, extractor: OpenLibraryExtractor
    ) -> None:
        payload = {
            "docs": [
                {"key": "/works/BAD", "title": "Bad", "subject": 42},
                {"key": "/works/GOOD", "title": "Good"},
            ]
        }
        respx.get(SEARCH).mock(return_value=httpx.Response(200, json=payload))

        items = await collect(extractor, limit=2)

        assert len([item for item in items if isinstance(item, Rejected)]) == 1
        assert len([item for item in items if isinstance(item, RawBook)]) == 1


class TestDumpShapedDocuments:
    """Open Library speaks two shapes and this mapper reads both.

    Search documents carry ``author_name``; dump editions carry author *keys*
    plus a ``by_statement``. Reading only the search shape silently dropped
    every author name the dump supplied — an end-to-end run loaded 5,890 books
    and zero authors before this was caught.
    """

    def test_a_by_statement_supplies_the_author(self) -> None:
        book = openlibrary.map_payload(
            {"key": "/books/OL1M", "title": "Antifa", "by_statement": "by Mark Bray"}
        )

        assert isinstance(book, RawBook)
        assert [a.name for a in book.authors] == ["Mark Bray"]

    @pytest.mark.parametrize(
        ("statement", "expected"),
        [
            ("by Mark Bray", "Mark Bray"),
            ("par Victor Hugo", "Victor Hugo"),
            ("von Franz Kafka", "Franz Kafka"),
            ("Mark Bray", "Mark Bray"),
            ("by Mark Bray.", "Mark Bray"),
            ("  by  Mark Bray  ", "Mark Bray"),
        ],
    )
    def test_the_leading_preposition_is_stripped(self, statement: str, expected: str) -> None:
        # The preposition is noise once the name is a search term.
        book = openlibrary.map_payload(
            {"key": "/books/OL1M", "title": "T", "by_statement": statement}
        )

        assert isinstance(book, RawBook)
        assert book.authors[0].name == expected

    def test_author_name_still_wins_when_present(self) -> None:
        # A search document's structured field beats prose.
        book = openlibrary.map_payload(
            {
                "key": "/works/OL1W",
                "title": "T",
                "author_name": ["Structured Name"],
                "by_statement": "by Prose Name",
            }
        )

        assert isinstance(book, RawBook)
        assert book.authors[0].name == "Structured Name"

    @pytest.mark.parametrize("statement", ["", "   ", "by ", None, 42])
    def test_an_unusable_by_statement_yields_no_author(self, statement: object) -> None:
        book = openlibrary.map_payload(
            {"key": "/books/OL1M", "title": "T", "by_statement": statement}
        )

        assert isinstance(book, RawBook)
        assert book.authors == []


class TestDumpFieldShapes:
    """Open Library's two shapes share almost no field names.

    Search returns isbn / first_publish_year / publisher / language; a dump
    edition returns isbn_13 / publish_date / publishers / languages. Reading
    only the search shape produced 6,000 title-only books in an end-to-end run
    — no ISBN, no year, no publisher, no subject between them.
    """

    @staticmethod
    def dump_edition(**fields: object) -> dict[str, object]:
        base: dict[str, object] = {
            "key": "/books/OL1M",
            "title": "Index to the House of Lords Debates",
            "isbn_13": ["9780107716837"],
            "isbn_10": ["0107716836"],
            "publish_date": "December 31, 1996",
            "publishers": ["Stationery Office Books"],
            "number_of_pages": 8,
            "subjects": ["Parliament", "Debates"],
            "languages": [{"key": "/languages/eng"}],
        }
        return base | fields

    def test_dump_isbns_are_read(self) -> None:
        book = openlibrary.map_payload(self.dump_edition())

        assert isinstance(book, RawBook)
        assert "9780107716837" in book.isbns
        assert "0107716836" in book.isbns

    def test_the_edition_publish_date_is_read(self) -> None:
        # Free text, handed to transform's parse_year rather than parsed twice.
        book = openlibrary.map_payload(self.dump_edition())

        assert isinstance(book, RawBook)
        assert book.published == "December 31, 1996"

    def test_a_search_year_still_wins_when_present(self) -> None:
        # A work's first_publish_year is the more specific claim.
        book = openlibrary.map_payload(self.dump_edition(first_publish_year=1965))

        assert isinstance(book, RawBook)
        assert book.published == "1965"

    def test_dump_publishers_are_read(self) -> None:
        book = openlibrary.map_payload(self.dump_edition())

        assert isinstance(book, RawBook)
        assert book.publisher == "Stationery Office Books"

    def test_dump_page_count_is_read(self) -> None:
        book = openlibrary.map_payload(self.dump_edition())

        assert isinstance(book, RawBook)
        assert book.page_count == 8

    def test_dump_subjects_are_read(self) -> None:
        book = openlibrary.map_payload(self.dump_edition())

        assert isinstance(book, RawBook)
        assert "Parliament" in book.subjects

    def test_dump_languages_are_unwrapped_from_their_keys(self) -> None:
        # [{"key": "/languages/eng"}] rather than ["eng"].
        book = openlibrary.map_payload(self.dump_edition())

        assert isinstance(book, RawBook)
        assert book.languages == ["eng"]

    def test_the_search_shape_is_unaffected(self) -> None:
        book = openlibrary.map_payload(
            {
                "key": "/works/OL1W",
                "title": "Dune",
                "isbn": ["9780441172719"],
                "first_publish_year": 1965,
                "publisher": ["Ace"],
                "language": ["eng"],
                "subject": ["Science fiction"],
                "number_of_pages_median": 412,
            }
        )

        assert isinstance(book, RawBook)
        assert book.isbns == ["9780441172719"]
        assert book.published == "1965"
        assert book.publisher == "Ace"
        assert book.languages == ["eng"]
        assert book.page_count == 412

    def test_a_record_with_neither_shape_still_maps(self) -> None:
        book = openlibrary.map_payload({"key": "/books/OL9M", "title": "Bare"})

        assert isinstance(book, RawBook)
        assert book.isbns == []
        assert book.published is None
