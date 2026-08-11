"""Open Library extractor: bounded metadata enrichment.

Open Library's developer guidance discourages bulk harvesting through the API
and points at data dumps instead, so this is deliberately not the bulk source.
It runs identified, at one request per second, with a hard record budget and an
explicit field list — without ``fields`` the search endpoint returns a thin
document and the enrichment is not worth the request.

What it adds that Gutendex cannot: ISBNs, publication years, publishers, page
counts, and stable work and author keys.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from itertools import zip_longest
from typing import Any

import structlog
from pydantic import ValidationError

from pipeline.config import Settings
from pipeline.extract.base import (
    DEFAULT_BASE_DELAY_SECONDS,
    ExtractedItem,
    ExtractionRequest,
    Rejected,
    Sleep,
    SourceUnavailableError,
    TokenBucket,
    build_client,
    request_with_retries,
)
from pipeline.models.domain import RawAuthor, RawBook, SourceName

logger = structlog.get_logger(__name__)

# Requested explicitly: the default search document omits most of these, and a
# request that comes back without them has cost the source a query for nothing.
FIELDS = (
    "key,title,subtitle,author_name,author_key,first_publish_year,isbn,"
    "language,number_of_pages_median,publisher,subject,cover_i"
)

DEFAULT_QUERY = "subject:fiction"
MAX_RESULTS_PER_PAGE = 100
COVER_URL = "https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"


class OpenLibraryExtractor:
    """Fetches bounded enrichment documents from Open Library search."""

    source_name = SourceName.OPENLIBRARY

    def __init__(
        self,
        settings: Settings,
        *,
        base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
        sleep: Sleep | None = None,
        page_size: int = MAX_RESULTS_PER_PAGE,
    ) -> None:
        self._settings = settings
        self._base_delay = base_delay
        self._sleep = sleep
        self._page_size = min(page_size, MAX_RESULTS_PER_PAGE)
        self._bucket = TokenBucket(settings.openlibrary_requests_per_second, sleep=sleep)

    async def fetch(self, request: ExtractionRequest) -> AsyncIterator[ExtractedItem]:
        """Yield records and rejections, one polite page at a time."""
        url = f"{self._settings.openlibrary_base_url.rstrip('/')}/search.json"
        query = request.query or DEFAULT_QUERY
        emitted = 0
        page = 1

        async with build_client(
            user_agent=self._settings.user_agent(),
            connect_timeout=self._settings.http_connect_timeout_seconds,
            read_timeout=self._settings.http_read_timeout_seconds,
        ) as client:
            while emitted < request.max_records:
                page_size = min(self._page_size, request.max_records - emitted)
                # The limiter guards the source, so it is acquired before the
                # request rather than slept after it.
                await self._bucket.acquire()
                response = await request_with_retries(
                    client,
                    "GET",
                    url,
                    params={
                        "q": query,
                        "fields": FIELDS,
                        "page": page,
                        "limit": page_size,
                    },
                    max_attempts=self._settings.http_max_attempts,
                    base_delay=self._base_delay,
                    sleep=self._sleep,
                    source=self.source_name.value,
                )
                try:
                    body = response.json()
                except json.JSONDecodeError as error:
                    raise SourceUnavailableError(
                        self.source_name.value,
                        f"response was not JSON: {error}",
                        status_code=response.status_code,
                    ) from error

                docs = body.get("docs") or []
                if not docs:
                    return

                for doc in docs:
                    if emitted >= request.max_records:
                        return
                    yield self._to_item(doc)
                    emitted += 1

                # A short page is the last page; another request would only
                # spend this source's one-per-second budget on nothing.
                if len(docs) < page_size:
                    return

                page += 1

    def _to_item(self, doc: dict[str, Any]) -> ExtractedItem:
        """Map one search document; any failure becomes a rejection."""
        key = doc.get("key")
        try:
            return RawBook(
                source=self.source_name,
                source_id=key,  # type: ignore[arg-type]
                title=doc.get("title"),  # type: ignore[arg-type]
                subtitle=doc.get("subtitle"),
                authors=_to_authors(doc),
                subjects=list(doc.get("subject") or [])[:50],
                isbns=list(doc.get("isbn") or []),
                languages=list(doc.get("language") or []),
                published=_year_as_text(doc.get("first_publish_year")),
                publisher=_first(doc.get("publisher")),
                page_count=doc.get("number_of_pages_median"),
                cover_url=(
                    COVER_URL.format(cover_id=doc["cover_i"]) if doc.get("cover_i") else None
                ),
                raw_payload=doc,
            )
        except ValidationError as error:
            logger.warning("openlibrary.record_rejected", source_id=key, errors=error.error_count())
            return Rejected(
                source=self.source_name,
                source_id=str(key) if key is not None else None,
                raw_payload=doc,
                rejection_code="invalid_record",
                detail="; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()
                )[:500],
            )


def _to_authors(doc: dict[str, Any]) -> list[RawAuthor]:
    """Zip the parallel name and key arrays.

    They are positionally paired but not guaranteed to be the same length, and
    a wrong pairing silently attributes a book to the wrong person — so the
    longer array wins and missing keys become None rather than shifting
    everything by one.
    """
    names = doc.get("author_name") or []
    keys = doc.get("author_key") or []
    return [
        RawAuthor(name=name, source_author_id=key) for name, key in zip_longest(names, keys) if name
    ]


def _first(values: list[str] | None) -> str | None:
    return values[0] if values else None


def _year_as_text(year: int | None) -> str | None:
    """Keep the year as the string transform expects.

    Parsing lives in transform, which already handles ``c1997`` and friends;
    doing it here would split year handling across two layers.
    """
    return str(year) if year is not None else None
