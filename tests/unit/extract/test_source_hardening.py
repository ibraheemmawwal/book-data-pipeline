"""Malformed source data across every adapter.

Public APIs return whatever they return. The contract is that a structurally
wrong record costs that record, and a structurally wrong *response* fails the
source cleanly — never an AttributeError halfway through a page, and never a
silently half-populated book.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pipeline.config import Settings
from pipeline.extract import goodreads_parsers, googlebooks, gutendex, openlibrary
from pipeline.extract.base import (
    ExtractionRequest,
    InvalidSourceRecordError,
    Rejected,
    SourceUnavailableError,
    TokenBucket,
    optional_list,
    optional_object,
    require_object,
    string_list,
)
from pipeline.extract.goodreads import map_payload as goodreads_map_payload
from pipeline.extract.googlebooks import GoogleBooksExtractor
from pipeline.extract.gutendex import GutendexExtractor
from pipeline.extract.openlibrary import OpenLibraryExtractor
from pipeline.models.domain import RawBook

GUTENDEX = "https://gutendex.com/books"
OL_SEARCH = "https://openlibrary.org/search.json"
GB_VOLUMES = "https://www.googleapis.com/books/v1/volumes"


class TestShapeHelpers:
    def test_require_object_rejects_a_scalar(self) -> None:
        with pytest.raises(InvalidSourceRecordError, match="record"):
            require_object("a string", "record")

    def test_require_object_rejects_a_list(self) -> None:
        with pytest.raises(InvalidSourceRecordError):
            require_object([1, 2], "record")

    def test_require_object_accepts_a_mapping(self) -> None:
        assert require_object({"a": 1}, "record") == {"a": 1}

    @pytest.mark.parametrize("value", ["a string", 42, {"not": "a list"}])
    def test_optional_list_rejects_a_non_list(self, value: object) -> None:
        with pytest.raises(InvalidSourceRecordError, match="authors"):
            optional_list({"authors": value}, "authors")

    def test_optional_list_treats_absent_and_null_as_empty(self) -> None:
        assert optional_list({}, "authors") == []
        assert optional_list({"authors": None}, "authors") == []

    def test_optional_object_rejects_a_non_object(self) -> None:
        with pytest.raises(InvalidSourceRecordError, match="formats"):
            optional_object({"formats": ["a", "list"]}, "formats")

    def test_optional_object_treats_absent_and_null_as_empty(self) -> None:
        assert optional_object({}, "formats") == {}
        assert optional_object({"formats": None}, "formats") == {}

    def test_string_list_rejects_a_non_string_member(self) -> None:
        # A subject list containing an object would otherwise reach the
        # database as the repr of a dict.
        with pytest.raises(InvalidSourceRecordError, match="only strings"):
            string_list({"subjects": ["fine", {"nested": "object"}]}, "subjects")

    def test_string_list_accepts_all_strings(self) -> None:
        assert string_list({"subjects": ["a", "b"]}, "subjects") == ["a", "b"]


class TestTokenBucketGuards:
    @pytest.mark.parametrize("rate", [0, -1.0])
    def test_a_non_positive_rate_is_refused(self, rate: float) -> None:
        # A zero rate would divide by zero and stall the extractor forever.
        with pytest.raises(ValueError, match="rate_per_second"):
            TokenBucket(rate)


async def collect(extractor: Any, limit: int = 50) -> list[RawBook | Rejected]:
    return [item async for item in extractor.fetch(ExtractionRequest(max_records=limit))]


class TestGutendexHardening:
    @pytest.fixture
    def extractor(self, settings: Settings) -> GutendexExtractor:
        return GutendexExtractor(settings, base_delay=0.0)

    @respx.mock
    async def test_a_non_object_result_is_rejected_not_fatal(
        self, extractor: GutendexExtractor
    ) -> None:
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(
                200,
                json={"next": None, "results": ["a bare string", {"id": 1, "title": "Fine"}]},
            )
        )

        items = await collect(extractor)

        assert len([i for i in items if isinstance(i, RawBook)]) == 1
        assert len([i for i in items if isinstance(i, Rejected)]) == 1

    @respx.mock
    async def test_a_non_object_results_container_fails_the_source(
        self, extractor: GutendexExtractor
    ) -> None:
        # A structurally wrong response is the source's problem, not one
        # record's, so it must not be silently treated as an empty page.
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": "not a list"})
        )

        with pytest.raises(SourceUnavailableError):
            await collect(extractor)

    @respx.mock
    async def test_a_top_level_array_fails_the_source(self, extractor: GutendexExtractor) -> None:
        respx.get(GUTENDEX).mock(return_value=httpx.Response(200, json=[1, 2, 3]))

        with pytest.raises(SourceUnavailableError):
            await collect(extractor)

    def test_a_non_object_payload_maps_to_a_rejection(self) -> None:
        assert isinstance(gutendex.map_payload("not an object"), Rejected)

    def test_a_bad_author_entry_rejects_only_its_record(self) -> None:
        rejected = gutendex.map_payload({"id": 1, "title": "T", "authors": ["a string"]})

        assert isinstance(rejected, Rejected)
        assert rejected.source_id == "1"


class TestOpenLibraryHardening:
    @pytest.fixture
    def extractor(self, settings: Settings) -> OpenLibraryExtractor:
        async def no_wait(_: float) -> None:
            return None

        return OpenLibraryExtractor(settings, base_delay=0.0, sleep=no_wait)

    @respx.mock
    async def test_a_non_json_body_fails_the_source(self, extractor: OpenLibraryExtractor) -> None:
        respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, text="<html>oops</html>"))

        with pytest.raises(SourceUnavailableError, match="not JSON"):
            await collect(extractor)

    @respx.mock
    async def test_a_non_list_docs_container_fails_the_source(
        self, extractor: OpenLibraryExtractor
    ) -> None:
        respx.get(OL_SEARCH).mock(
            return_value=httpx.Response(200, json={"docs": {"not": "a list"}})
        )

        with pytest.raises(SourceUnavailableError):
            await collect(extractor)

    @respx.mock
    async def test_a_non_numeric_publish_year_rejects_the_document(
        self, extractor: OpenLibraryExtractor
    ) -> None:
        respx.get(OL_SEARCH).mock(
            return_value=httpx.Response(
                200,
                json={
                    "docs": [
                        {"key": "/works/OL1W", "title": "Bad", "first_publish_year": {"y": 1}},
                        {"key": "/works/OL2W", "title": "Fine"},
                    ]
                },
            )
        )

        items = await collect(extractor, limit=2)

        assert len([i for i in items if isinstance(i, Rejected)]) == 1
        assert len([i for i in items if isinstance(i, RawBook)]) == 1

    @respx.mock
    async def test_a_non_integer_cover_id_rejects_the_document(
        self, extractor: OpenLibraryExtractor
    ) -> None:
        respx.get(OL_SEARCH).mock(
            return_value=httpx.Response(
                200,
                json={"docs": [{"key": "/works/OL1W", "title": "T", "cover_i": "abc"}]},
            )
        )

        items = await collect(extractor, limit=1)

        assert isinstance(items[0], Rejected)

    def test_a_year_supplied_as_a_string_is_accepted(self) -> None:
        # Open Library is inconsistent about this and both forms are valid.
        book = openlibrary.map_payload(
            {"key": "/works/OL1W", "title": "T", "first_publish_year": "1965"}
        )

        assert isinstance(book, RawBook)
        assert book.published == "1965"


class TestGoogleBooksHardening:
    @pytest.fixture
    def extractor(self, settings: Settings) -> GoogleBooksExtractor:
        return GoogleBooksExtractor(settings, base_delay=0.0)

    @respx.mock
    async def test_a_non_json_body_fails_the_source(self, extractor: GoogleBooksExtractor) -> None:
        respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, text="not json"))

        with pytest.raises(SourceUnavailableError, match="not JSON"):
            await collect(extractor)

    @respx.mock
    async def test_a_non_list_items_container_fails_the_source(
        self, extractor: GoogleBooksExtractor
    ) -> None:
        respx.get(GB_VOLUMES).mock(
            return_value=httpx.Response(200, json={"items": {"not": "a list"}})
        )

        with pytest.raises(SourceUnavailableError):
            await collect(extractor)

    @pytest.mark.parametrize(
        "identifiers",
        [
            [{"type": 13, "identifier": "9780553380163"}],
            [{"type": "ISBN_13", "identifier": 9780553380163}],
            [{"type": "ISBN_13", "identifier": ""}],
        ],
    )
    def test_a_malformed_industry_identifier_rejects_the_volume(
        self, identifiers: list[dict[str, Any]]
    ) -> None:
        # A number where an ISBN string belongs would become a canonical
        # identity built from the repr of an int.
        rejected = googlebooks.map_payload(
            {"id": "g1", "volumeInfo": {"title": "T", "industryIdentifiers": identifiers}}
        )

        assert isinstance(rejected, Rejected)

    def test_a_non_string_image_link_rejects_the_volume(self) -> None:
        rejected = googlebooks.map_payload(
            {"id": "g1", "volumeInfo": {"title": "T", "imageLinks": {"thumbnail": 42}}}
        )

        assert isinstance(rejected, Rejected)

    def test_a_non_string_language_rejects_the_volume(self) -> None:
        rejected = googlebooks.map_payload(
            {"id": "g1", "volumeInfo": {"title": "T", "language": ["en"]}}
        )

        assert isinstance(rejected, Rejected)

    def test_a_non_object_volume_info_rejects_the_volume(self) -> None:
        assert isinstance(googlebooks.map_payload({"id": "g1", "volumeInfo": "nope"}), Rejected)

    def test_a_non_object_payload_maps_to_a_rejection(self) -> None:
        assert isinstance(googlebooks.map_payload(["not", "an", "object"]), Rejected)


class TestGoodreadsHardening:
    def test_an_unparseable_series_position_drops_only_the_position(self) -> None:
        # The relationship is still real even when the number is nonsense.
        book = goodreads_map_payload({"bookId": "1", "title": "Book (Series, #not-a-number)"})

        assert isinstance(book, RawBook)

    def test_json_ld_authors_accepts_an_object(self) -> None:
        assert goodreads_parsers.json_ld_authors({"name": "Frank Herbert"}) == ["Frank Herbert"]

    def test_json_ld_authors_accepts_an_array(self) -> None:
        # JSON-LD is inconsistent here and both shapes appear in the wild.
        assert goodreads_parsers.json_ld_authors([{"name": "A"}, {"name": "B"}]) == ["A", "B"]

    def test_json_ld_authors_accepts_bare_strings(self) -> None:
        assert goodreads_parsers.json_ld_authors(["A", "  B  "]) == ["A", "B"]

    def test_json_ld_authors_skips_unusable_entries(self) -> None:
        assert goodreads_parsers.json_ld_authors([{"nope": 1}, "", 42, {"name": "C"}]) == ["C"]

    def test_a_non_numeric_rating_is_dropped_not_fatal(self) -> None:
        book = goodreads_map_payload({"bookId": "1", "title": "X", "avgRating": "n/a"})

        assert isinstance(book, RawBook)
        assert book.goodreads_average_rating is None

    def test_an_out_of_range_rating_is_dropped(self) -> None:
        # Goodreads ratings are 0-5; anything else is a contract change.
        book = goodreads_map_payload({"bookId": "1", "title": "X", "avgRating": "9.9"})

        assert isinstance(book, RawBook)
        assert book.goodreads_average_rating is None


class TestPaginationHardening:
    @respx.mock
    async def test_gutendex_a_non_string_next_link_fails_the_source(
        self, settings: Settings
    ) -> None:
        # Following a non-URL would either crash or silently end the walk; both
        # are worse than saying the response was malformed.
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(
                200, json={"next": {"page": 2}, "results": [{"id": 1, "title": "T"}]}
            )
        )
        extractor = GutendexExtractor(settings, base_delay=0.0)

        with pytest.raises(SourceUnavailableError, match="next"):
            await collect(extractor)

    @respx.mock
    async def test_googlebooks_stops_mid_page_at_the_budget(self, settings: Settings) -> None:
        # The limit is a budget, not a filter: a page richer than the remaining
        # budget must be truncated rather than overspent.
        volumes = {
            "items": [{"id": f"g{n}", "volumeInfo": {"title": f"Book {n}"}} for n in range(5)]
        }
        respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json=volumes))
        extractor = GoogleBooksExtractor(settings, base_delay=0.0, page_size=5)

        assert len(await collect(extractor, limit=2)) == 2

    @respx.mock
    async def test_openlibrary_stops_mid_page_at_the_budget(self, settings: Settings) -> None:
        async def no_wait(_: float) -> None:
            return None

        docs = {"docs": [{"key": f"/works/OL{n}W", "title": f"B{n}"} for n in range(5)]}
        respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json=docs))
        extractor = OpenLibraryExtractor(settings, base_delay=0.0, sleep=no_wait, page_size=5)

        assert len(await collect(extractor, limit=2)) == 2


class TestNonObjectResponses:
    """A top-level array or scalar where an object belongs.

    Distinct from a malformed field: the whole response shape is wrong, so
    there is no record to reject and the source fails cleanly instead.
    """

    @respx.mock
    async def test_googlebooks_rejects_a_top_level_array(self, settings: Settings) -> None:
        respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json=[1, 2, 3]))

        with pytest.raises(SourceUnavailableError, match="response"):
            await collect(GoogleBooksExtractor(settings, base_delay=0.0))

    @respx.mock
    async def test_openlibrary_rejects_a_top_level_array(self, settings: Settings) -> None:
        async def no_wait(_: float) -> None:
            return None

        respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json=["nope"]))

        with pytest.raises(SourceUnavailableError, match="response"):
            await collect(OpenLibraryExtractor(settings, base_delay=0.0, sleep=no_wait))

    @respx.mock
    async def test_googlebooks_rejects_a_top_level_scalar(self, settings: Settings) -> None:
        respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json="a string"))

        with pytest.raises(SourceUnavailableError):
            await collect(GoogleBooksExtractor(settings, base_delay=0.0))
