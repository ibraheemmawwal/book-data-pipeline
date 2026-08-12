"""Resolving candidates at once without asking any source faster than agreed.

Concurrency here is only defensible because each source's published rate is
enforced across the whole run. Before this, the rate limiter was rebuilt for
every call and so enforced nothing — sequential resolution was the only thing
keeping the pipeline polite, which meant politeness was a property of the loop
shape rather than of the code claiming to provide it.

These tests fail if that regresses, which is the point: the failure mode is
invisible locally and looks like a block from the source days later.
"""

from __future__ import annotations

import asyncio
from itertools import pairwise

import httpx
import pytest
import respx
from pydantic import SecretStr

from pipeline.config import Settings
from pipeline.extract.goodreads import GoodreadsExtractor
from pipeline.extract.openlibrary import OpenLibraryExtractor
from pipeline.extract.resolver import CatalogueResolver
from pipeline.ingest import _resolve_batch
from pipeline.models.domain import CandidateBook, SourceName

OL_SEARCH = "https://openlibrary.org/search.json"
GB_VOLUMES = "https://www.googleapis.com/books/v1/volumes"
GUTENDEX = "https://gutendex.com/books"


@pytest.fixture
def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        database_url="postgresql+psycopg://u:p@localhost/db",
        openlibrary_contact_email="t@example.com",
    )


def candidates(count: int) -> list[CandidateBook]:
    return [CandidateBook(candidate_key=f"/books/OL{n}M", title=f"Book {n}") for n in range(count)]


class TestTheRateLimiterOutlivesOneCall:
    def test_one_extractor_serves_the_whole_run(self, settings: Settings) -> None:
        """The invariant the limiter depends on.

        A TokenBucket only limits anything if it is asked twice. Rebuilding the
        extractor per attempt handed every call a bucket with nothing to
        remember.
        """
        resolver = CatalogueResolver(settings)
        first = resolver._extractor_for(SourceName.OPENLIBRARY)
        second = resolver._extractor_for(SourceName.OPENLIBRARY)

        assert first is second

    @respx.mock
    async def test_requests_are_spaced_even_when_candidates_overlap(
        self, settings: Settings
    ) -> None:
        """The actual guarantee: one request per second to Open Library.

        Time is driven by a fake clock and a fake sleep, so this asserts the
        spacing the limiter computes rather than how fast the machine runs.
        """
        now = 0.0
        arrivals: list[float] = []

        def clock() -> float:
            return now

        async def sleep(seconds: float) -> None:
            nonlocal now
            now += seconds

        respx.get(OL_SEARCH).mock(
            side_effect=lambda _r: (
                arrivals.append(now),
                httpx.Response(200, json={"docs": []}),
            )[1]
        )
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )
        respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json={"totalItems": 0}))

        resolver = CatalogueResolver(settings)
        # One shared extractor, with the fake clock wired into its limiter.
        resolver._extractors[SourceName.OPENLIBRARY] = OpenLibraryExtractor(
            settings, sleep=sleep, base_delay=0.0
        )
        resolver._extractors[SourceName.OPENLIBRARY]._bucket._monotonic = clock  # type: ignore[attr-defined]
        resolver._extractors[SourceName.OPENLIBRARY]._bucket._sleep = sleep  # type: ignore[attr-defined]

        await _resolve_batch(resolver, candidates(4), concurrency=4)

        assert len(arrivals) == 4
        gaps = [b - a for a, b in pairwise(arrivals)]
        # 1.0 is the published Open Library rate. Concurrency must overlap the
        # waiting, never the asking.
        assert all(gap >= 1.0 for gap in gaps), f"requests {gaps} apart, faster than agreed"


class TestConcurrencyIsBounded:
    @respx.mock
    async def test_no_more_than_the_limit_are_in_flight(self, settings: Settings) -> None:
        in_flight = 0
        peak = 0

        async def watch(_request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1
            return httpx.Response(200, json={"docs": []})

        respx.get(OL_SEARCH).mock(side_effect=watch)
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )
        respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json={"totalItems": 0}))
        fast = settings.model_copy(update={"openlibrary_requests_per_second": 1000.0})
        resolver = CatalogueResolver(fast)

        await _resolve_batch(resolver, candidates(10), concurrency=3)

        assert peak <= 3, f"{peak} candidates in flight against a limit of 3"

    @respx.mock
    async def test_every_candidate_still_gets_a_result(self, settings: Settings) -> None:
        # Concurrency must not drop or reorder work.
        respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json={"docs": []}))
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )
        respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json={"totalItems": 0}))
        fast = settings.model_copy(update={"openlibrary_requests_per_second": 1000.0})
        batch = candidates(6)

        results = await _resolve_batch(CatalogueResolver(fast), batch, concurrency=4)

        assert [r.candidate.candidate_key for r in results] == [c.candidate_key for c in batch], (
            "results came back out of order; the run's accounting pairs them with the batch"
        )


class TestGoodreadsStaysOneAtATime:
    @respx.mock
    async def test_only_one_goodreads_request_is_ever_in_flight(self, settings: Settings) -> None:
        """The rule the old sequential loop existed to protect.

        It is kept, but paid for by Goodreads alone rather than by every other
        source in the run.
        """
        in_flight = 0
        peak = 0

        async def watch(_request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1
            return httpx.Response(200, json=[])

        respx.route(host="www.goodreads.com").mock(side_effect=watch)
        respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json={"docs": []}))
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )
        respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json={"totalItems": 0}))

        gated = settings.model_copy(
            update={
                "goodreads_enabled": True,
                "goodreads_unofficial_source_accepted": True,
                "goodreads_in_resolution": True,
                "goodreads_requests_per_second": 1000.0,
                "openlibrary_requests_per_second": 1000.0,
                "googlebooks_api_key": SecretStr("k"),
            }
        )
        resolver = CatalogueResolver(gated, goodreads=GoodreadsExtractor(gated))

        await _resolve_batch(resolver, candidates(6), concurrency=6)

        assert peak <= 1, f"{peak} concurrent Goodreads requests; it permits one"
