"""Gutendex extractor: the primary bulk source.

Gutendex mirrors Project Gutenberg. It publishes no ISBN, publisher, page count
or publication year, which shapes the whole catalogue: the ISBN-less identity
path is the common case rather than an edge case, and year-based analytics have
almost nothing to work with. What it does carry densely is author lifespan,
subjects and download counts, so those are mapped as first-class fields.

Pagination follows the ``next`` link the API returns rather than constructing
page numbers, because the link is the source's own statement about where the
next page is and a constructed URL is our guess about it.
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

COVER_MIME = "image/jpeg"


class GutendexExtractor:
    """Fetches book records from a Gutendex instance."""

    source_name = SourceName.GUTENDEX

    def __init__(
        self, settings: Settings, *, base_delay: float = DEFAULT_BASE_DELAY_SECONDS
    ) -> None:
        self._settings = settings
        self._base_delay = base_delay

    async def fetch(self, request: ExtractionRequest) -> AsyncIterator[ExtractedItem]:
        """Yield records and rejections, following ``next`` until the budget runs out."""
        url: str | None = f"{self._settings.gutendex_base_url.rstrip('/')}/books"
        params: dict[str, Any] | None = {"page": 1}
        emitted = 0

        async with build_client(
            user_agent=self._settings.user_agent(),
            connect_timeout=self._settings.http_connect_timeout_seconds,
            read_timeout=self._settings.http_read_timeout_seconds,
        ) as client:
            while url is not None and emitted < request.max_records:
                response = await request_with_retries(
                    client,
                    "GET",
                    url,
                    params=params,
                    max_attempts=self._settings.http_max_attempts,
                    base_delay=self._base_delay,
                    source=self.source_name.value,
                )
                try:
                    page = response.json()
                except json.JSONDecodeError as error:
                    # An HTML error page behind a 200 is a source problem, not a
                    # programming one, so it fails the source rather than the run.
                    raise SourceUnavailableError(
                        self.source_name.value,
                        f"response was not JSON: {error}",
                        status_code=response.status_code,
                    ) from error

                for record in page.get("results", []):
                    if emitted >= request.max_records:
                        return
                    yield self._to_item(record)
                    emitted += 1

                # The API's own next link; params are already baked into it.
                url = page.get("next")
                params = None

    def _to_item(self, record: dict[str, Any]) -> ExtractedItem:
        """Map one source record, turning any failure into a rejection.

        Never raises. One malformed record must cost that record, not the page.
        """
        source_id = record.get("id")
        try:
            return RawBook(
                source=self.source_name,
                source_id=source_id,  # type: ignore[arg-type]
                title=record.get("title"),  # type: ignore[arg-type]
                authors=[_to_author(a) for a in record.get("authors", [])],
                subjects=list(record.get("subjects", [])),
                language=_first(record.get("languages")),
                description=_first(record.get("summaries")),
                cover_url=(record.get("formats") or {}).get(COVER_MIME),
                download_count=record.get("download_count"),
                raw_payload=record,
            )
        except ValidationError as error:
            logger.warning(
                "gutendex.record_rejected",
                source_id=source_id,
                errors=error.error_count(),
            )
            return Rejected(
                source=self.source_name,
                source_id=str(source_id) if source_id is not None else None,
                raw_payload=record,
                rejection_code="invalid_record",
                detail="; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()
                )[:500],
            )


def _to_author(payload: dict[str, Any]) -> RawAuthor:
    return RawAuthor(
        name=payload.get("name"),  # type: ignore[arg-type]
        birth_year=payload.get("birth_year"),
        death_year=payload.get("death_year"),
    )


def _first(values: list[str] | None) -> str | None:
    """The first entry, or None. Gutendex uses lists even for single values."""
    return values[0] if values else None
