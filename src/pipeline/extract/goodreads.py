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
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
import structlog
from pydantic import ValidationError

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
    parse_aria_series,
    parse_book_detail,
    parse_first_edition,
    parse_series_from_title,
    score_candidate,
    split_title_by_author,
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

    async def resolve(
        self, client: httpx.AsyncClient, query: str, *, isbn: str | None = None
    ) -> RawBook | None:
        """Resolve one candidate: autocomplete, rank, then enrich with detail.

        ISBN queries bypass ranking entirely. An ISBN is an exact identifier,
        so Goodreads' own candidate ordering is better evidence than string
        similarity against a title we may already have wrong.

        Detail is fetched for the top-ranked candidate only, falling through to
        the second and third if its pages cannot be parsed. Fanning detail
        requests across every autocomplete result would multiply our traffic
        against an unofficial source for no gain.

        Returns ``None`` when nothing matches. Never raises for a miss — that
        is the resolver's cue to fall back, not an error.
        """
        cache_key = f"isbn:{isbn}" if isbn else f"q:{query.casefold()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        candidates = await self.autocomplete(client, isbn or query)
        if not candidates:
            return None

        ranked = self._rank(candidates, query, isbn)
        if not ranked:
            return None

        for candidate in ranked[:MAX_DETAIL_ATTEMPTS]:
            observation = self.to_raw_book(candidate)
            if isinstance(observation, Rejected):
                continue
            enriched = await self._enrich(client, observation, candidate)
            # Only validated, non-empty results are cached. Caching a miss
            # would turn a transient blip into a run-long hole.
            self._cache.put(cache_key, enriched, is_isbn=isbn is not None)
            return enriched

        return None

    def _rank(
        self, candidates: list[dict[str, Any]], query: str, isbn: str | None
    ) -> list[dict[str, Any]]:
        """Order candidates best-first, or trust the source for ISBN queries."""
        if isbn:
            return candidates

        title, author = split_title_by_author(query)
        scored = [
            (
                score_candidate(
                    title,
                    author,
                    str(candidate.get("bookTitleBare") or candidate.get("title") or ""),
                    (candidate.get("author") or {}).get("name")
                    if isinstance(candidate.get("author"), dict)
                    else None,
                ),
                index,
                candidate,
            )
            for index, candidate in enumerate(candidates)
        ]
        # Index breaks ties so ordering is total and the same response always
        # ranks the same way.
        return [
            candidate
            for score, _, candidate in sorted(scored, key=lambda item: (-item[0], item[1]))
            if score >= self._settings.goodreads_min_match_score
        ]

    async def _enrich(
        self,
        client: httpx.AsyncClient,
        observation: RawBook,
        candidate: dict[str, Any],
    ) -> RawBook:
        """Add detail-page facts to an autocomplete observation.

        The book and work pages are fetched concurrently with
        ``return_exceptions=True``: one surviving page still produces a better
        observation than neither, and detail is enrichment rather than a
        precondition. Autocomplete alone is already a valid record.
        """
        book_id = candidate.get("bookId")
        work_id = candidate.get("workId")
        if not book_id:
            return observation

        paths = [self._get(client, f"/book/show/{book_id}")]
        if work_id:
            paths.append(self._get(client, f"/work/editions/{work_id}"))

        results = await asyncio.gather(*paths, return_exceptions=True)
        book_html = results[0] if isinstance(results[0], str) else None
        work_html = results[1] if len(results) > 1 and isinstance(results[1], str) else None

        updates: dict[str, Any] = {}
        payload = dict(observation.raw_payload)

        if book_html is not None:
            detail = parse_book_detail(book_html)
            payload["_detail"] = detail.payload
            if detail.description:
                updates["description"] = detail.description
            if detail.series is not None:
                updates["series"] = [detail.series]
            if detail.page_count:
                updates["page_count"] = detail.page_count

        if work_html is not None:
            edition = parse_first_edition(work_html)
            payload["_edition"] = edition.payload
            if edition.isbn13:
                updates["isbns"] = [edition.isbn13]
            if edition.published:
                updates["published"] = edition.published
            if edition.publisher:
                updates["publisher"] = edition.publisher

        if not updates:
            return observation

        updates["raw_payload"] = payload
        return observation.model_copy(update=updates)

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

        # Enrichment is replayed, not re-fetched. The load layer rebuilds
        # canonical fields from stored payloads, so anything the detail pages
        # contributed must be reconstructible from what was saved — otherwise a
        # confirmed series quietly downgrades to an inferred one on re-ingest.
        detail = _stored_block(candidate, "_detail")
        edition = _stored_block(candidate, "_edition")
        series = _replayed_series(detail) or series

        return RawBook(
            source=SOURCE,
            source_id=book_id,  # type: ignore[arg-type]
            title=bare,
            isbns=[edition["isbn13"]] if isinstance(edition.get("isbn13"), str) else [],
            published=edition.get("published"),
            publisher=edition.get("publisher"),
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


def _stored_block(candidate: dict[str, Any], field: str) -> dict[str, Any]:
    """An enrichment block saved by ``_enrich``, or an empty mapping."""
    block = candidate.get(field)
    return block if isinstance(block, dict) else {}


def _replayed_series(detail: dict[str, Any]) -> list[RawSeriesMembership]:
    """Rebuild a series relationship from a stored detail block.

    The ARIA label and the confirmed series id are what the detail page
    contributed; without replaying them, a re-ingest would downgrade an
    evidenced relationship back to an inferred one.
    """
    label = detail.get("series_label")
    parsed = parse_aria_series(label if isinstance(label, str) else None)
    if parsed is None:
        return []

    name, position = parsed
    series_id = detail.get("series_id")
    return [
        RawSeriesMembership(
            name=name,
            source_series_id=series_id if isinstance(series_id, str) else None,
            position=str(position) if position is not None else None,
            confirmed=isinstance(series_id, str) and bool(series_id),
        )
    ]
