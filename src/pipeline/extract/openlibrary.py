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
    InvalidSourceRecordError,
    Rejected,
    Sleep,
    SourceUnavailableError,
    TokenBucket,
    build_client,
    optional_list,
    record_error_detail,
    request_with_retries,
    require_object,
    string_list,
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
                    before_attempt=self._bucket.acquire,
                    source=self.source_name.value,
                )
                try:
                    body = require_object(response.json(), "response")
                except json.JSONDecodeError as error:
                    raise SourceUnavailableError(
                        self.source_name.value,
                        f"response was not JSON: {error}",
                        status_code=response.status_code,
                    ) from error
                except InvalidSourceRecordError as error:
                    raise SourceUnavailableError(
                        self.source_name.value,
                        str(error),
                        status_code=response.status_code,
                    ) from error

                try:
                    docs = optional_list(body, "docs")
                except InvalidSourceRecordError as error:
                    raise SourceUnavailableError(
                        self.source_name.value,
                        str(error),
                        status_code=response.status_code,
                    ) from error
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

    def _to_item(self, payload: object) -> ExtractedItem:
        """Map one search document; any failure becomes a rejection."""
        key: object = None
        try:
            doc = require_object(payload, "document")
            key = doc.get("key")
            cover_id = doc.get("cover_i")
            if cover_id is not None and (
                isinstance(cover_id, bool) or not isinstance(cover_id, int)
            ):
                msg = f"cover_i must be an integer, got {type(cover_id).__name__}"
                raise InvalidSourceRecordError(msg)
            return RawBook(
                source=self.source_name,
                source_id=key,  # type: ignore[arg-type]
                title=doc.get("title"),  # type: ignore[arg-type]
                subtitle=doc.get("subtitle"),
                authors=_to_authors(doc),
                subjects=string_list(doc, "subject")[:50],
                isbns=string_list(doc, "isbn"),
                languages=string_list(doc, "language"),
                published=_year_as_text(doc.get("first_publish_year")),
                publisher=_first_non_empty(string_list(doc, "publisher")),
                page_count=doc.get("number_of_pages_median"),
                cover_url=COVER_URL.format(cover_id=cover_id) if cover_id else None,
                raw_payload=doc,
            )
        except (InvalidSourceRecordError, ValidationError) as error:
            logger.warning(
                "openlibrary.record_rejected",
                source_id=key,
                errors=error.error_count() if isinstance(error, ValidationError) else 1,
            )
            return Rejected(
                source=self.source_name,
                source_id=str(key) if key is not None else None,
                raw_payload=payload,
                rejection_code="invalid_record",
                detail=record_error_detail(error),
            )


def _to_authors(doc: dict[str, Any]) -> list[RawAuthor]:
    """Zip the parallel name and key arrays.

    They are positionally paired but not guaranteed to be the same length, and
    a wrong pairing silently attributes a book to the wrong person — so the
    longer array wins and missing keys become None rather than shifting
    everything by one.
    """
    names = string_list(doc, "author_name")
    keys = string_list(doc, "author_key")
    return [
        RawAuthor(name=name, source_author_id=key) for name, key in zip_longest(names, keys) if name
    ]


def _first_non_empty(values: list[str]) -> str | None:
    return next((value for value in values if value.strip()), None)


def _year_as_text(year: object) -> str | None:
    """Keep the year as the string transform expects.

    Parsing lives in transform, which already handles ``c1997`` and friends;
    doing it here would split year handling across two layers.
    """
    if year is None:
        return None
    if isinstance(year, bool) or not isinstance(year, (int, str)):
        msg = f"first_publish_year must be an integer or string, got {type(year).__name__}"
        raise InvalidSourceRecordError(msg)
    return str(year)
