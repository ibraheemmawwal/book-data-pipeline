"""Google Books extractor.

Credential-gated. A clean clone has no API key, so the absence of one must
produce a recorded skip rather than a crash or a silent no-op — that is what
keeps `docker compose up` working for someone who just cloned the repo.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from pipeline.config import Settings
from pipeline.extract.base import ExtractionRequest, Rejected, SourceUnavailableError
from pipeline.extract.googlebooks import GoogleBooksExtractor, MissingCredentialError
from pipeline.models.domain import RawBook, SourceName

from .conftest import load_fixture

VOLUMES = "https://www.googleapis.com/books/v1/volumes"


async def collect(extractor: GoogleBooksExtractor, limit: int = 100) -> list[RawBook | Rejected]:
    return [item async for item in extractor.fetch(ExtractionRequest(max_records=limit))]


@pytest.fixture
def extractor(settings: Settings) -> GoogleBooksExtractor:
    return GoogleBooksExtractor(settings, base_delay=0.0)


@pytest.fixture
def keyless(settings: Settings) -> Settings:
    return settings.model_copy(update={"googlebooks_api_key": None})


class TestCredentialGating:
    async def test_without_a_key_it_refuses_to_start(self, keyless: Settings) -> None:
        extractor = GoogleBooksExtractor(keyless, base_delay=0.0)

        with pytest.raises(MissingCredentialError):
            await collect(extractor)

    async def test_the_refusal_explains_itself_for_source_runs(self, keyless: Settings) -> None:
        # This string is written to source_runs.error, so a skipped source is
        # visible in the run record rather than inferred from a gap.
        extractor = GoogleBooksExtractor(keyless, base_delay=0.0)

        with pytest.raises(MissingCredentialError) as caught:
            await collect(extractor)

        assert "api key" in str(caught.value).lower()

    @respx.mock
    async def test_without_a_key_no_request_is_ever_made(self, keyless: Settings) -> None:
        route = respx.get(VOLUMES).mock(return_value=httpx.Response(200, json={}))
        extractor = GoogleBooksExtractor(keyless, base_delay=0.0)

        with pytest.raises(MissingCredentialError):
            await collect(extractor)

        assert not route.called

    async def test_the_skip_is_distinguishable_from_a_source_outage(
        self, keyless: Settings
    ) -> None:
        # A missing credential is our misconfiguration; an outage is theirs.
        # Conflating them makes source_runs useless for triage.
        extractor = GoogleBooksExtractor(keyless, base_delay=0.0)

        with pytest.raises(MissingCredentialError) as caught:
            await collect(extractor)

        assert not isinstance(caught.value, SourceUnavailableError)


class TestCredentialHandling:
    @respx.mock
    async def test_the_key_is_sent_as_a_query_parameter(
        self, extractor: GoogleBooksExtractor
    ) -> None:
        route = respx.get(VOLUMES).mock(
            return_value=httpx.Response(200, json=load_fixture("googlebooks_volumes.json"))
        )
        await collect(extractor, limit=2)

        assert route.calls[0].request.url.params["key"] == "test-key"

    @respx.mock
    async def test_the_key_never_appears_in_a_log_or_error(
        self, extractor: GoogleBooksExtractor
    ) -> None:
        respx.get(VOLUMES).mock(return_value=httpx.Response(500))

        with pytest.raises(SourceUnavailableError) as caught:
            await collect(extractor)

        assert "test-key" not in str(caught.value)


class TestMapping:
    @respx.mock
    async def test_maps_a_volume(self, extractor: GoogleBooksExtractor) -> None:
        respx.get(VOLUMES).mock(
            return_value=httpx.Response(200, json=load_fixture("googlebooks_volumes.json"))
        )

        books = [b for b in await collect(extractor) if isinstance(b, RawBook)]
        hawking = books[0]

        assert hawking.source is SourceName.GOOGLEBOOKS
        assert hawking.source_id == "hVj4CwAAQBAJ"
        assert hawking.title == "A Brief History of Time"
        assert hawking.subtitle == "From the Big Bang to Black Holes"
        assert hawking.publisher == "Bantam"
        assert hawking.page_count == 212
        assert hawking.published == "1998-09-01"
        assert hawking.language == "en"

    @respx.mock
    async def test_collects_both_isbn_forms(self, extractor: GoogleBooksExtractor) -> None:
        # industryIdentifiers mixes ISBN_10 and ISBN_13; transform decides which
        # to canonicalise, so extraction keeps both.
        respx.get(VOLUMES).mock(
            return_value=httpx.Response(200, json=load_fixture("googlebooks_volumes.json"))
        )

        books = [b for b in await collect(extractor) if isinstance(b, RawBook)]

        assert set(books[0].isbns) == {"9780553380163", "0553380168"}

    @respx.mock
    async def test_ignores_non_isbn_industry_identifiers(
        self, extractor: GoogleBooksExtractor
    ) -> None:
        payload = {
            "totalItems": 1,
            "items": [
                {
                    "id": "x1",
                    "volumeInfo": {
                        "title": "Odd identifiers",
                        "industryIdentifiers": [
                            {"type": "OTHER", "identifier": "internal-42"},
                            {"type": "ISSN", "identifier": "1234-5678"},
                            {"type": "ISBN_13", "identifier": "9780553380163"},
                        ],
                    },
                }
            ],
        }
        respx.get(VOLUMES).mock(return_value=httpx.Response(200, json=payload))

        books = [b for b in await collect(extractor) if isinstance(b, RawBook)]

        assert books[0].isbns == ["9780553380163"]

    @respx.mock
    async def test_prefers_the_larger_thumbnail(self, extractor: GoogleBooksExtractor) -> None:
        respx.get(VOLUMES).mock(
            return_value=httpx.Response(200, json=load_fixture("googlebooks_volumes.json"))
        )

        books = [b for b in await collect(extractor) if isinstance(b, RawBook)]

        assert books[0].cover_url is not None
        assert "zoom=1" in books[0].cover_url

    @respx.mock
    async def test_a_sparse_volume_still_maps(self, extractor: GoogleBooksExtractor) -> None:
        # Most real volumes lack description, pageCount and imageLinks.
        respx.get(VOLUMES).mock(
            return_value=httpx.Response(200, json=load_fixture("googlebooks_volumes.json"))
        )

        books = [b for b in await collect(extractor) if isinstance(b, RawBook)]
        dune = books[1]

        assert dune.title == "Dune"
        assert dune.description is None
        assert dune.page_count is None
        assert dune.cover_url is None

    @respx.mock
    async def test_carries_no_download_count(self, extractor: GoogleBooksExtractor) -> None:
        respx.get(VOLUMES).mock(
            return_value=httpx.Response(200, json=load_fixture("googlebooks_volumes.json"))
        )

        books = [b for b in await collect(extractor) if isinstance(b, RawBook)]

        assert all(b.download_count is None for b in books)


class TestPerItemIsolation:
    @respx.mock
    async def test_a_volume_without_a_title_is_rejected(
        self, extractor: GoogleBooksExtractor
    ) -> None:
        payload = {
            "totalItems": 2,
            "items": [
                {"id": "bad", "volumeInfo": {"authors": ["Nobody"]}},
                {"id": "good", "volumeInfo": {"title": "Fine"}},
            ],
        }
        respx.get(VOLUMES).mock(return_value=httpx.Response(200, json=payload))

        items = await collect(extractor)

        assert len([i for i in items if isinstance(i, RawBook)]) == 1
        rejects = [i for i in items if isinstance(i, Rejected)]
        assert len(rejects) == 1
        assert rejects[0].source_id == "bad"

    @respx.mock
    async def test_an_absent_items_key_is_an_empty_result_not_a_crash(
        self, extractor: GoogleBooksExtractor
    ) -> None:
        # Google omits `items` entirely on a zero-result query.
        respx.get(VOLUMES).mock(
            return_value=httpx.Response(200, json={"kind": "books#volumes", "totalItems": 0})
        )

        assert await collect(extractor) == []


class TestQuota:
    @respx.mock
    async def test_a_real_quota_response_is_retried_then_reported(
        self, extractor: GoogleBooksExtractor
    ) -> None:
        # The captured payload is the genuine 429 this project received when
        # the anonymous daily quota ran out.
        respx.get(VOLUMES).mock(
            return_value=httpx.Response(429, json=load_fixture("googlebooks_rate_limited.json"))
        )

        with pytest.raises(SourceUnavailableError) as caught:
            await collect(extractor)

        assert caught.value.status_code == 429
