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
    InvalidSourceRecordError,
    Rejected,
    SourceUnavailableError,
    build_client,
    optional_list,
    optional_object,
    record_error_detail,
    request_with_retries,
    require_object,
    string_list,
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
                    page = require_object(response.json(), "response")
                except json.JSONDecodeError as error:
                    # An HTML error page behind a 200 is a source problem, not a
                    # programming one, so it fails the source rather than the run.
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
                    results = optional_list(page, "results")
                except InvalidSourceRecordError as error:
                    raise SourceUnavailableError(
                        self.source_name.value,
                        str(error),
                        status_code=response.status_code,
                    ) from error

                for record in results:
                    if emitted >= request.max_records:
                        return
                    yield self._to_item(record)
                    emitted += 1

                # The API's own next link; params are already baked into it.
                next_url = page.get("next")
                if next_url is not None and not isinstance(next_url, str):
                    raise SourceUnavailableError(
                        self.source_name.value,
                        f"next must be a URL string or null, got {type(next_url).__name__}",
                        status_code=response.status_code,
                    )
                url = next_url
                params = None

    def _to_item(self, payload: object) -> ExtractedItem:
        """Map one source record, turning any failure into a rejection.

        Never raises. One malformed record must cost that record, not the page.
        """
        record: dict[str, Any] | None = None
        source_id: object = None
        try:
            record = require_object(payload, "record")
            source_id = record.get("id")
            authors = [_to_author(author) for author in optional_list(record, "authors")]
            formats = optional_object(record, "formats")
            return RawBook(
                source=self.source_name,
                source_id=source_id,  # type: ignore[arg-type]
                title=record.get("title"),  # type: ignore[arg-type]
                authors=authors,
                subjects=string_list(record, "subjects"),
                languages=string_list(record, "languages"),
                description=_first_non_empty(string_list(record, "summaries")),
                cover_url=formats.get(COVER_MIME),
                download_count=record.get("download_count"),
                raw_payload=record,
            )
        except (InvalidSourceRecordError, ValidationError) as error:
            logger.warning(
                "gutendex.record_rejected",
                source_id=source_id,
                errors=error.error_count() if isinstance(error, ValidationError) else 1,
            )
            return Rejected(
                source=self.source_name,
                source_id=str(source_id) if source_id is not None else None,
                raw_payload=payload,
                rejection_code="invalid_record",
                detail=record_error_detail(error),
            )


def _to_author(payload: object) -> RawAuthor:
    document = require_object(payload, "authors item")
    return RawAuthor(
        name=document.get("name"),  # type: ignore[arg-type]
        birth_year=document.get("birth_year"),
        death_year=document.get("death_year"),
    )


def _first_non_empty(values: list[str]) -> str | None:
    """The first non-blank entry, or None."""
    return next((value for value in values if value.strip()), None)
