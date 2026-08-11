"""Goodreads adapter: the preferred resolver, on an unofficial contract.

Goodreads ended public API access in 2020. Autocomplete returns JSON, but book
and work-edition detail are HTML pages, and none of the three is a supported
interface. The owner has explicitly accepted that product, availability and
terms-of-use risk; see the ADR. This module's job is to make that risk
*contained* rather than pretended away.

Containment, all of it load-bearing:

- **Two gates.** Both ``goodreads_enabled`` and
  ``goodreads_unofficial_source_accepted`` must be true. A clean clone runs the
  documented-API path with neither.
- **Honest identity.** A real ``User-Agent`` naming the project and a contact.
  No browser impersonation, no proxy or fingerprint rotation.
- **No control bypass.** 401, 403 and challenge pages are treated as terminal
  for the circuit interval, never retried or worked around.
- **One request in flight**, at most five starts per second, hard five-second
  timeouts.
- **A circuit breaker** that stops the run's remaining candidates after
  repeated access or contract failures, so one upstream outage cannot become
  thousands of failing calls.
- **Cache only validated, non-empty results.** Caching an empty answer would
  turn a transient blip into a run-long hole in the catalogue.

Failure here is never fatal: the resolver falls back to documented APIs.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
import structlog
from pydantic import ValidationError
from selectolax.parser import HTMLParser

from pipeline.config import Settings
from pipeline.extract.base import (
    ExtractedItem,
    Rejected,
    Sleep,
    SourceUnavailableError,
    TokenBucket,
    build_client,
)
from pipeline.extract.goodreads_parsers import (
    clean_html_text,
    is_placeholder_cover,
    parse_series_from_title,
    upgrade_cover_url,
)
from pipeline.models.domain import RawAuthor, RawBook, RawSeriesMembership, SourceName

logger = structlog.get_logger(__name__)

SOURCE = SourceName.GOODREADS

# Access controls. Never retried and never worked around — treating a block as
# a transient error is how a polite integration becomes an abusive one.
ACCESS_DENIED_STATUS = frozenset({401, 403, 429})
CHALLENGE_MARKERS = ("captcha", "are you a robot", "unusual traffic", "cf-challenge")

MAX_DETAIL_ATTEMPTS = 3
MAX_RATING = Decimal(5)
SERVER_ERROR_THRESHOLD = 500
CLIENT_ERROR_THRESHOLD = 400
_ARIA_SERIES = re.compile(r"Book\s+([\d.]+)\s+in\s+the\s+(.+?)\s+series", re.IGNORECASE)
_SERIES_HREF = re.compile(r"/series/(\d+)(?:-([a-z0-9-]+))?", re.IGNORECASE)
_NON_SLUG = re.compile(r"[^a-z0-9]+")


class GoodreadsUnavailableError(SourceUnavailableError):
    """Goodreads could not be used for this candidate.

    Distinct from a generic source failure so the resolver can tell "fall back
    and keep going" from "this run is broken".
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(SOURCE.value, message, status_code=status_code)


class GoodreadsNotAcceptedError(Exception):
    """The unofficial-source risk has not been accepted in configuration."""


@dataclass
class _CircuitBreaker:
    """Stops hammering a source that is refusing us.

    Opened by repeated access failures or contract failures. Once open it stays
    open for the run: re-probing a site that has blocked us is exactly the
    behaviour the containment rules forbid.
    """

    threshold: int
    failures: int = 0
    opened: bool = False
    reason: str | None = None

    def record_failure(self, reason: str) -> None:
        self.failures += 1
        if self.failures >= self.threshold and not self.opened:
            self.opened = True
            self.reason = reason
            logger.warning("goodreads.circuit_open", reason=reason, failures=self.failures)

    def trip(self, reason: str) -> None:
        """Open immediately, for a failure that will not fix itself."""
        self.opened = True
        self.reason = reason
        logger.warning("goodreads.circuit_open", reason=reason, immediate=True)

    def record_success(self) -> None:
        self.failures = 0


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    value: RawBook
    expires_at: float


class GoodreadsResultCache:
    """Caches only validated, non-empty results.

    Empty results and failures are deliberately never cached: doing so would
    turn one transient blip into a run-long hole in the catalogue. ISBN lookups
    live longer than title lookups because an ISBN is an exact identifier whose
    answer does not drift.
    """

    def __init__(self, title_ttl: int, isbn_ttl: int) -> None:
        self._title_ttl = title_ttl
        self._isbn_ttl = isbn_ttl
        self._entries: dict[str, _CacheEntry] = {}

    def get(self, key: str, *, now: float | None = None) -> RawBook | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if (now if now is not None else time.monotonic()) >= entry.expires_at:
            del self._entries[key]
            return None
        return entry.value

    def put(self, key: str, value: RawBook, *, is_isbn: bool, now: float | None = None) -> None:
        ttl = self._isbn_ttl if is_isbn else self._title_ttl
        if ttl <= 0:
            return
        base = now if now is not None else time.monotonic()
        self._entries[key] = _CacheEntry(value=value, expires_at=base + ttl)


def _looks_like_challenge(body: str) -> bool:
    lowered = body[:4000].lower()
    return any(marker in lowered for marker in CHALLENGE_MARKERS)


def _slugify(value: str) -> str:
    return _NON_SLUG.sub("-", value.strip().casefold()).strip("-")


def parse_series_id(href: str | None, series_name: str) -> str | None:
    """Accept a Goodreads series id only when its slug matches the name.

    An id taken from an unrelated link would attach a book to the wrong series
    permanently, and nothing downstream could detect it. When the slug does not
    agree the relationship is still recorded — just unconfirmed.
    """
    if not href:
        return None
    match = _SERIES_HREF.search(href)
    if match is None:
        return None
    slug = match.group(2)
    if slug and _slugify(slug) != _slugify(series_name):
        return None
    return match.group(1)


def parse_aria_series(label: str | None) -> tuple[str, Decimal | None] | None:
    """Read ``Book 2.5 in the Discworld series`` from an ARIA label."""
    if not label:
        return None
    match = _ARIA_SERIES.search(label)
    if match is None:
        return None
    try:
        position: Decimal | None = Decimal(match.group(1))
    except InvalidOperation:
        position = None
    name = match.group(2).strip()
    return (name, position) if name else None


def parse_json_ld(html: str) -> dict[str, Any] | None:
    """Pull the Book JSON-LD block out of a detail page.

    Preferred over scraping attributes because it is structured data the site
    publishes deliberately, so it changes less often than the markup around it.
    """
    tree = HTMLParser(html)
    for node in tree.css('script[type="application/ld+json"]'):
        raw = node.text(strip=True)
        if not raw:
            continue
        try:
            document = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidates = document if isinstance(document, list) else [document]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("@type") in {"Book", "Product"}:
                return candidate
    return None


def json_ld_authors(value: Any) -> list[str]:
    """JSON-LD ``author`` is an object or an array of them, never reliably one."""
    entries = value if isinstance(value, list) else [value]
    names = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name"):
            names.append(str(entry["name"]).strip())
        elif isinstance(entry, str) and entry.strip():
            names.append(entry.strip())
    return names


class GoodreadsExtractor:
    """Resolves one candidate at a time against Goodreads."""

    source_name = SOURCE

    def __init__(
        self,
        settings: Settings,
        *,
        sleep: Sleep | None = None,
    ) -> None:
        self._settings = settings
        self._sleep = sleep
        self._bucket = TokenBucket(settings.goodreads_requests_per_second, sleep=sleep)
        # One in flight. Concurrency against an unofficial source is exactly
        # the behaviour that gets an integration blocked.
        self._gate = asyncio.Semaphore(settings.goodreads_max_in_flight)
        self._circuit = _CircuitBreaker(settings.goodreads_circuit_failure_threshold)
        self._cache = GoodreadsResultCache(
            settings.goodreads_title_cache_ttl_seconds,
            settings.goodreads_isbn_cache_ttl_seconds,
        )

    @property
    def circuit_open(self) -> bool:
        """Whether the resolver should stop trying Goodreads for this run."""
        return self._circuit.opened

    def ensure_accepted(self) -> None:
        """Refuse to run unless both gates are set.

        Raises:
            GoodreadsNotAcceptedError: enabling an unofficial source has to be
                a deliberate, separately-recorded acknowledgement.
        """
        if not self._settings.goodreads_enabled:
            raise GoodreadsNotAcceptedError("goodreads is disabled by configuration")
        if not self._settings.goodreads_unofficial_source_accepted:
            raise GoodreadsNotAcceptedError(
                "goodreads is enabled but the unofficial-source risk has not been "
                "accepted; set PIPELINE_GOODREADS_UNOFFICIAL_SOURCE_ACCEPTED=true"
            )

    async def _get(
        self, client: httpx.AsyncClient, path: str, params: dict[str, Any] | None = None
    ) -> str:
        """One rate-limited, single-in-flight request with a hard timeout."""
        if self._circuit.opened:
            raise GoodreadsUnavailableError(f"circuit open: {self._circuit.reason}")

        async with self._gate:
            await self._bucket.acquire()
            try:
                response = await client.get(path, params=params)
            except httpx.TransportError as error:
                self._circuit.record_failure(f"transport: {type(error).__name__}")
                raise GoodreadsUnavailableError(f"transport failure: {error}") from error

        if response.status_code in ACCESS_DENIED_STATUS:
            # A block is an answer, not an error to retry around.
            self._circuit.trip(f"access denied: HTTP {response.status_code}")
            raise GoodreadsUnavailableError(
                f"access denied (HTTP {response.status_code})",
                status_code=response.status_code,
            )
        if response.status_code >= SERVER_ERROR_THRESHOLD:
            self._circuit.record_failure(f"HTTP {response.status_code}")
            raise GoodreadsUnavailableError(
                f"upstream error HTTP {response.status_code}",
                status_code=response.status_code,
            )
        if response.status_code >= CLIENT_ERROR_THRESHOLD:
            raise GoodreadsUnavailableError(
                f"HTTP {response.status_code}", status_code=response.status_code
            )

        body = response.text
        if _looks_like_challenge(body):
            self._circuit.trip("challenge page returned")
            raise GoodreadsUnavailableError("challenge page returned")

        self._circuit.record_success()
        return body

    def build_client(self) -> httpx.AsyncClient:
        """A client identifying itself honestly, with a hard timeout."""
        timeout = self._settings.goodreads_timeout_seconds
        client = build_client(
            user_agent=self._settings.user_agent(),
            connect_timeout=timeout,
            read_timeout=timeout,
        )
        client.base_url = httpx.URL(self._settings.goodreads_base_url)
        return client

    async def autocomplete(self, client: httpx.AsyncClient, query: str) -> list[dict[str, Any]]:
        """Search titles. ``format=json`` is mandatory or the route returns HTML."""
        body = await self._get(client, "/book/auto_complete", {"format": "json", "q": query})
        try:
            results = json.loads(body)
        except json.JSONDecodeError as error:
            self._circuit.record_failure("autocomplete was not JSON")
            raise GoodreadsUnavailableError(
                f"autocomplete response was not JSON: {error}"
            ) from error
        return [item for item in results if isinstance(item, dict)]

    def to_raw_book(self, candidate: dict[str, Any]) -> ExtractedItem:
        return _candidate_to_raw_book(candidate)


def _to_rating(value: Any) -> Decimal | None:
    """Autocomplete sends the rating as a string, sometimes absent."""
    if value in (None, ""):
        return None
    try:
        rating = Decimal(str(value))
    except InvalidOperation:
        return None
    return rating if Decimal(0) <= rating <= MAX_RATING else None


def _candidate_to_raw_book(candidate: dict[str, Any]) -> ExtractedItem:
    """Map one autocomplete candidate into a RawBook.

    Autocomplete alone carries title, author, page count, rating, cover and
    the series embedded in the dirty title — enough for a valid observation
    without a detail request. Detail enriches it; it is not a precondition.
    """
    book_id = candidate.get("bookId")
    try:
        dirty_title = str(candidate.get("title") or "")
        parsed_series = parse_series_from_title(dirty_title)
        bare = candidate.get("bookTitleBare") or (
            parsed_series.bare_title if parsed_series else dirty_title
        )

        author = candidate.get("author") or {}
        authors = (
            [
                RawAuthor(
                    name=str(author["name"]),
                    source_author_id=str(author["id"]) if author.get("id") else None,
                )
            ]
            if isinstance(author, dict) and author.get("name")
            else []
        )

        cover = upgrade_cover_url(candidate.get("imageUrl"))
        description = candidate.get("description")
        if isinstance(description, dict):
            description = description.get("html")

        series = []
        if parsed_series is not None:
            series.append(
                RawSeriesMembership(
                    name=parsed_series.name,
                    position=str(parsed_series.position)
                    if parsed_series.position is not None
                    else None,
                    # Inferred from title text, not confirmed by a /series/
                    # link. Detail parsing can upgrade this.
                    confirmed=False,
                )
            )

        return RawBook(
            source=SOURCE,
            source_id=book_id,  # type: ignore[arg-type]
            title=bare,
            authors=authors,
            description=clean_html_text(description),
            page_count=candidate.get("numPages") or None,
            cover_url=None if is_placeholder_cover(cover) else cover,
            goodreads_average_rating=_to_rating(candidate.get("avgRating")),
            series=series,
            raw_payload=candidate,
        )
    except (ValidationError, TypeError, ValueError) as error:
        logger.warning("goodreads.record_rejected", source_id=book_id)
        return Rejected(
            source=SOURCE,
            source_id=str(book_id) if book_id is not None else None,
            raw_payload=candidate,
            rejection_code="invalid_record",
            detail=str(error)[:500],
        )


def map_payload(payload: object) -> ExtractedItem:
    """Re-map a stored Goodreads payload.

    The load layer recomputes canonical fields by replaying provenance, so this
    must work with no client, no settings and no network — which it does,
    because an autocomplete candidate is self-contained.
    """
    if not isinstance(payload, dict):
        return Rejected(
            source=SOURCE,
            source_id=None,
            raw_payload={},
            rejection_code="invalid_record",
            detail=f"expected an object, got {type(payload).__name__}",
        )
    return _candidate_to_raw_book(payload)
