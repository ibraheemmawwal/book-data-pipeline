"""Google Books extractor: credential-gated enrichment and conflict input.

The key is optional by design. A clean clone has none, and the pipeline must
still start — so a missing credential raises a distinct error the DAG records
as a skip in ``source_runs`` rather than a failure. Conflating "we are not
configured for this source" with "this source is down" makes the run record
useless for triage, which is why ``MissingCredentialError`` is deliberately not
a ``SourceUnavailableError``.

Google Books is the only source that reliably carries publisher, page count and
a precise publication date, which makes it the useful third opinion when two
sources disagree.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import structlog
from pydantic import ValidationError

from pipeline.config import Settings
from pipeline.extract.base import (
    DEFAULT_BASE_DELAY_SECONDS,
    ExtractedItem,
    ExtractionRequest,
    Rejected,
    SourceUnavailableError,
    build_client,
    request_with_retries,
)
from pipeline.models.domain import RawAuthor, RawBook, SourceName

logger = structlog.get_logger(__name__)

ISBN_IDENTIFIER_TYPES = frozenset({"ISBN_10", "ISBN_13"})
MAX_RESULTS_PER_PAGE = 40
DEFAULT_QUERY = "subject:fiction"


class MissingCredentialError(Exception):
    """The source is enabled but has no API key.

    Not a ``SourceUnavailableError``: this is our configuration gap, not the
    provider's outage, and the two need different entries in ``source_runs``.
    """


class GoogleBooksExtractor:
    """Fetches volumes from the Google Books API."""

    source_name = SourceName.GOOGLEBOOKS

    def __init__(
        self,
        settings: Settings,
        *,
        base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
        page_size: int = MAX_RESULTS_PER_PAGE,
    ) -> None:
        self._settings = settings
        self._base_delay = base_delay
        self._page_size = min(page_size, MAX_RESULTS_PER_PAGE)

    async def fetch(self, request: ExtractionRequest) -> AsyncIterator[ExtractedItem]:
        """Yield records and rejections.

        Raises:
            MissingCredentialError: no API key is configured. Raised before any
                network call, so an unconfigured source costs nothing.
            SourceUnavailableError: the source failed terminally.
        """
        secret = self._settings.googlebooks_api_key
        if secret is None:
            raise MissingCredentialError(
                "googlebooks is enabled but no API key is configured; "
                "set PIPELINE_GOOGLEBOOKS_API_KEY or disable the source"
            )

        url = f"{self._settings.googlebooks_base_url.rstrip('/')}/books/v1/volumes"
        query = request.query or DEFAULT_QUERY
        emitted = 0

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
                        "startIndex": emitted,
                        "maxResults": page_size,
                        "key": secret.get_secret_value(),
                    },
                    max_attempts=self._settings.http_max_attempts,
                    base_delay=self._base_delay,
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

                # Google omits `items` entirely rather than sending an empty list.
                items = body.get("items") or []
                if not items:
                    return

                for volume in items:
                    if emitted >= request.max_records:
                        return
                    yield self._to_item(volume)
                    emitted += 1

                # A page shorter than the one requested is the last page.
                # Waiting for an empty one costs an extra round trip, and
                # against a source that ignores startIndex it never arrives.
                if len(items) < page_size:
                    return

    def _to_item(self, volume: dict[str, Any]) -> ExtractedItem:
        """Map one volume; any failure becomes a rejection."""
        volume_id = volume.get("id")
        info = volume.get("volumeInfo") or {}
        try:
            return RawBook(
                source=self.source_name,
                source_id=volume_id,  # type: ignore[arg-type]
                title=info.get("title"),  # type: ignore[arg-type]
                subtitle=info.get("subtitle"),
                authors=[RawAuthor(name=n) for n in info.get("authors") or [] if n],
                subjects=list(info.get("categories") or []),
                isbns=_isbns(info),
                language=info.get("language"),
                published=info.get("publishedDate"),
                publisher=info.get("publisher"),
                page_count=info.get("pageCount"),
                description=info.get("description"),
                cover_url=_cover(info),
                raw_payload=volume,
            )
        except ValidationError as error:
            logger.warning(
                "googlebooks.record_rejected",
                source_id=volume_id,
                errors=error.error_count(),
            )
            return Rejected(
                source=self.source_name,
                source_id=str(volume_id) if volume_id is not None else None,
                raw_payload=volume,
                rejection_code="invalid_record",
                detail="; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()
                )[:500],
            )


def _isbns(info: dict[str, Any]) -> list[str]:
    """Pull ISBNs out of industryIdentifiers.

    The array also carries OTHER and ISSN entries; treating those as ISBNs
    would poison canonical identity with values that are not book numbers.
    """
    return [
        identifier["identifier"]
        for identifier in info.get("industryIdentifiers") or []
        if identifier.get("type") in ISBN_IDENTIFIER_TYPES and identifier.get("identifier")
    ]


def _cover(info: dict[str, Any]) -> str | None:
    """The largest cover Google offers in a search response."""
    links = info.get("imageLinks") or {}
    return links.get("thumbnail") or links.get("smallThumbnail")
