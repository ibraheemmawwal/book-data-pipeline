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
- **One request in flight**, one start every two seconds, hard five-second
  timeouts.
- **A circuit breaker** that stops the run's remaining candidates after
  repeated access or contract failures, so one upstream outage cannot become
  thousands of failing calls.
- **A cooldown that outlives the run.** The breaker is per-process and every
  scheduled task is a fresh process, so a refusal is written to ``source_runs``
  and every Goodreads path checks it before opening a client. Without it a run
  refused at 14:17 is rediscovered hourly, and a sequence of individually
  correct runs behaves like a retry loop. See ``pipeline.source_health``.
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
from enum import Enum, auto
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
    BookDetail,
    clean_html_text,
    is_placeholder_cover,
    is_plausible_isbn_match,
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

# The work editions page is HTML and 404s if asked for JSON.
HTML_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
MAX_RATING = Decimal(5)
SERVER_ERROR_THRESHOLD = 500
# First retry waits this long, then it doubles. On top of the rate limiter's
# own spacing, so three retries span roughly fourteen seconds.
TRANSIENT_BACKOFF_SECONDS = 1.0
# A block's first wait, and the factor it grows by: 10s, then 60s, then 300s
# (capped by goodreads_block_pause_seconds). Most blocks clear inside the first
# of those; the rare stubborn one needs the last.
BLOCK_BACKOFF_SECONDS = 10.0
BLOCK_BACKOFF_FACTOR = 6.0


class _Verdict(Enum):
    """What a response means for whether to ask again, and how soon."""

    OK = auto()
    TRANSIENT = auto()
    BLOCKED = auto()


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
    """Stops hammering a source that is refusing us, or that is failing.

    Opened by repeated access failures or contract failures. Once open it stays
    open for the run: re-probing a site that has blocked us is exactly the
    behaviour the containment rules forbid.

    ``refused`` separates the two reasons it can open, because they call for
    different responses and conflating them cost us a working pipeline. A 403,
    a challenge page or a run of empty bodies is Goodreads making a decision
    about *us*, and the right answer is to stay away for a while. A 503 is
    Goodreads failing to serve anyone, and the right answer is to come back
    shortly — measured live, one request in three returned 503 while the other
    two returned a complete 150KB page to the same client. Treating that as a
    refusal opened the circuit, and then paused every DAG for ninety minutes,
    over a source that was answering us.
    """

    threshold: int
    failures: int = 0
    opened: bool = False
    reason: str | None = None
    refused: bool = False

    def record_failure(self, reason: str, *, refusal: bool) -> None:
        self.failures += 1
        if self.failures >= self.threshold and not self.opened:
            self.opened = True
            self.reason = reason
            self.refused = refusal
            logger.warning(
                "goodreads.circuit_open", reason=reason, failures=self.failures, refusal=refusal
            )

    def trip(self, reason: str, *, refusal: bool = True) -> None:
        """Open immediately, for a failure that will not fix itself."""
        self.opened = True
        self.reason = reason
        self.refused = refusal
        logger.warning("goodreads.circuit_open", reason=reason, immediate=True, refusal=refusal)

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


def _looks_empty(body: str) -> bool:
    """Whether a 2xx returned nothing at all.

    Observed live: the same book page that served 849KB minutes earlier began
    answering 202 with zero bytes once the day's requests added up. Treating
    that as a successful fetch is how a run reports resolutions it never made —
    the parser finds no fields, the record degrades to a thin one, and nothing
    says the source stopped talking to us.

    Empty rather than merely small, deliberately. A legitimate answer can be
    tiny — an autocomplete miss is two characters — and a threshold generous
    enough to cover a book page would reject those.
    """
    return not body.strip()


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
        self._transient_retries = settings.goodreads_transient_retries
        self._block_retries = settings.goodreads_block_retries
        self._block_pause = settings.goodreads_block_pause_seconds
        self._cache = GoodreadsResultCache(
            settings.goodreads_title_cache_ttl_seconds,
            settings.goodreads_isbn_cache_ttl_seconds,
        )

    def _block_wait(self, attempt: int) -> float:
        """How long to wait out a block on its ``attempt``-th consecutive hit.

        Escalating, because "blocked" covers two very different events and a
        single figure serves neither. Measured at five seconds between
        requests, the first was refused and the next succeeded — most of these
        clear in seconds. Measured on a bad one, probing a page a minute, it
        took between four and five minutes.

        A flat five-minute pause paid the worst case every time and turned a
        blip into a stalled slice. Ten seconds, then a minute, then five gets
        the common case back almost immediately and still outlasts the rare
        one, for six minutes of worst-case waiting rather than ten.
        """
        return min(self._block_pause, BLOCK_BACKOFF_SECONDS * BLOCK_BACKOFF_FACTOR ** (attempt - 1))

    async def _wait(self, seconds: float) -> None:
        """Sleep, through the injected clock when there is one."""
        if self._sleep is not None:
            await self._sleep(seconds)
        else:
            await asyncio.sleep(seconds)

    async def _backoff(self, attempt: int) -> None:
        """Wait before retrying a transient failure.

        Doubling, on top of the two seconds the rate limiter already imposes,
        so a source under load is not asked again at the same cadence that
        found it under load.
        """
        await self._wait(TRANSIENT_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    @property
    def circuit_open(self) -> bool:
        """Whether the resolver should stop trying Goodreads for this run."""
        return self._circuit.opened

    @property
    def refused(self) -> bool:
        """Whether the source made a decision about us, rather than just failing.

        Only this warrants the cross-run cooldown. A run stopped by upstream
        5xx should end and let the next one try; a run stopped by a block
        should keep every other DAG away too.
        """
        return self._circuit.opened and self._circuit.refused

    @property
    def circuit_reason(self) -> str | None:
        """Why the circuit opened, for a report that explains itself."""
        return self._circuit.reason

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
        self,
        client: httpx.AsyncClient,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        accept: str | None = None,
    ) -> str:
        """One rate-limited, single-in-flight request with a hard timeout.

        ``accept`` overrides the client default per request, which this source
        genuinely requires: its bot mitigation answers the same client
        differently depending on the header. The shared client asks for JSON,
        which the autocomplete and book endpoints answer with 200 — and which
        the work editions page answers with **404**. Not a redirect, not a
        block, a plain 404 for a page that exists and returns 125KB of HTML the
        moment the header changes.

        That single header cost every publication year in the catalogue:
        Goodreads publishes no year anywhere except that page, so 480 records
        carried a workId, asked for its editions, were told the page did not
        exist, and moved on without complaint.
        """
        if self._circuit.opened:
            raise GoodreadsUnavailableError(f"circuit open: {self._circuit.reason}")

        # Two failure kinds, two budgets, because they are different events.
        #
        # A transient failure is Goodreads failing to serve anyone — 503 to
        # about one request in three at present, in clusters — and a few
        # seconds of backoff absorbs it.
        #
        # A block is Goodreads refusing everyone with 202 and an empty body,
        # and it is global: the minute these records returned nothing, so did
        # Dune and The Catcher in the Rye. Moving to the next book cannot help.
        # But it lifts on its own and quickly — probing one page a minute, it
        # cleared between the fourth and fifth — so it is waited out inside the
        # run rather than ending it.
        transient = 0
        blocked = 0
        while True:
            async with self._gate:
                await self._bucket.acquire()
                try:
                    response = await client.get(
                        path, params=params, headers={"Accept": accept} if accept else None
                    )
                except httpx.TransportError as error:
                    if transient >= self._transient_retries:
                        self._circuit.record_failure(
                            f"transport: {type(error).__name__}", refusal=False
                        )
                        raise GoodreadsUnavailableError(f"transport failure: {error}") from error
                    transient += 1
                    await self._backoff(transient)
                    continue

            kind, reason = self._classify(response)
            if kind is _Verdict.OK:
                self._circuit.record_success()
                return response.text

            if kind is _Verdict.BLOCKED:
                if blocked >= self._block_retries:
                    self._circuit.record_failure(reason, refusal=True)
                    raise GoodreadsUnavailableError(reason, status_code=response.status_code)
                blocked += 1
                pause = self._block_wait(blocked)
                logger.warning(
                    "goodreads.blocked_waiting",
                    path=path,
                    pause_seconds=pause,
                    attempt=blocked,
                    of=self._block_retries,
                )
                await self._wait(pause)
                continue

            if transient >= self._transient_retries:
                self._circuit.record_failure(reason, refusal=False)
                raise GoodreadsUnavailableError(reason, status_code=response.status_code)
            transient += 1
            logger.info("goodreads.transient_retry", path=path, attempt=transient, reason=reason)
            await self._backoff(transient)

    def _classify(self, response: httpx.Response) -> tuple[_Verdict, str]:
        """What this response is, or raise if it settles the matter.

        Raising here rather than returning a third kind keeps the caller's loop
        to the two things it can actually act on: wait a little, or wait a lot.

        Raises:
            GoodreadsUnavailableError: the response is final — a block we must
                not retry around, a challenge page, or a 4xx about this page.
        """
        if response.status_code in ACCESS_DENIED_STATUS:
            # An answer, not an error to retry around.
            self._circuit.trip(f"access denied: HTTP {response.status_code}")
            raise GoodreadsUnavailableError(
                f"access denied (HTTP {response.status_code})",
                status_code=response.status_code,
            )
        if response.status_code >= SERVER_ERROR_THRESHOLD:
            return _Verdict.TRANSIENT, f"HTTP {response.status_code}"
        if response.status_code >= CLIENT_ERROR_THRESHOLD:
            # A 404 is about this page, not about us: never retried, never
            # counted, because repeating it would only confirm that the book
            # does not exist.
            raise GoodreadsUnavailableError(
                f"HTTP {response.status_code}", status_code=response.status_code
            )
        if _looks_like_challenge(response.text):
            self._circuit.trip("challenge page returned")
            raise GoodreadsUnavailableError("challenge page returned")
        if _looks_empty(response.text):
            return _Verdict.BLOCKED, f"empty body on HTTP {response.status_code}"
        return _Verdict.OK, ""

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
            if enriched is None:
                # The fall-through this loop always documented and never did:
                # it returned the un-enriched card instead, so candidates two
                # and three were never reached.
                continue
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
            # Goodreads' ordering stands, but a result that shares almost
            # nothing with the title we asked for means the two providers
            # disagree about what this ISBN denotes. Guessing which is right is
            # worse than falling back to a documented source.
            title, _ = split_title_by_author(query)
            kept = [
                candidate
                for candidate in candidates
                if is_plausible_isbn_match(
                    title,
                    str(candidate.get("bookTitleBare") or candidate.get("title") or ""),
                )
            ]
            if len(kept) != len(candidates):
                logger.info(
                    "goodreads.isbn_mismatch_discarded",
                    isbn=isbn,
                    discarded=len(candidates) - len(kept),
                )
            return kept

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
    ) -> RawBook | None:
        """Add detail-page facts to an autocomplete observation.

        Returns ``None`` when neither detail page yielded anything, so the
        caller can try the next candidate instead of keeping a card.

        A search card is not a record of a book. It carries no publication year
        — Goodreads publishes none outside the editions page — and a single
        author where the work may have three. Accepting one anyway is how 480
        stored observations came to have 10.4% detail coverage and no years
        between them, each one looking like a successful resolution.

        The two pages are still fetched with ``return_exceptions=True`` and
        either alone is enough: they fail independently, and the site answers
        them differently. What is no longer acceptable is neither.
        """
        book_id = candidate.get("bookId")
        work_id = candidate.get("workId")
        if not book_id:
            return None

        # Deliberately different Accept headers. The book page returns 200 for
        # a JSON ask and 202 with an empty challenge body for a browser-like
        # one; the editions page does the reverse. Neither is documented and
        # both were established by measurement.
        paths = [self._get(client, f"/book/show/{book_id}")]
        if work_id:
            paths.append(self._get(client, f"/work/editions/{work_id}", accept=HTML_ACCEPT))

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
            if detail.authors:
                # The full credited list. The search card names one author, so
                # without this a book with three keeps only the first.
                updates["authors"] = [RawAuthor(name=name) for name in detail.authors]

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
            logger.warning(
                "goodreads.detail_empty",
                book_id=book_id,
                work_id=work_id,
                detail="neither detail page yielded anything; trying the next candidate",
            )
            return None

        updates["raw_payload"] = payload
        return observation.model_copy(update=updates)

    async def enrich_by_id(self, client: httpx.AsyncClient, observation: RawBook) -> RawBook | None:
        """Complete a record we already hold a Goodreads id for.

        An export gives a title, its authors and a rating; it gives no year, no
        ISBN, no page count and no series. Those live on the book's own page,
        so a book id has to become two requests to be worth anything.

        The work id is not in the export either — it is only on the book page,
        as the link to the editions list. So the order is forced: fetch the
        book page, learn the work id, and only then decide whether the editions
        page is worth a second request. It is worth one when the book page
        withheld an ISBN or a year, and not otherwise: most books give both up
        first time, and asking anyway would double the traffic against a source
        that has asked not to be crawled for the sake of the minority.
        """
        book_id = observation.source_id
        if not book_id:
            return None

        try:
            book_html = await self._get(client, f"/book/show/{book_id}")
        except GoodreadsUnavailableError:
            return None

        detail = parse_book_detail(book_html)
        payload = dict(observation.raw_payload)
        payload["_detail"] = detail.payload
        updates = _updates_from_detail(detail)

        if detail.work_id and not (detail.isbn and detail.published):
            try:
                work_html = await self._get(
                    client, f"/work/editions/{detail.work_id}", accept=HTML_ACCEPT
                )
            except GoodreadsUnavailableError:
                work_html = None
            if work_html is not None:
                edition = parse_first_edition(work_html)
                payload["_edition"] = edition.payload
                if edition.isbn13 and "isbns" not in updates:
                    updates["isbns"] = [edition.isbn13]
                if edition.published and "published" not in updates:
                    updates["published"] = edition.published
                if edition.publisher:
                    updates["publisher"] = edition.publisher

        if not updates:
            logger.warning("goodreads.enrich_by_id_empty", book_id=book_id)
            return None

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
            self._circuit.record_failure("autocomplete was not JSON", refusal=False)
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


def _updates_from_detail(detail: BookDetail) -> dict[str, Any]:
    """The fields a book page contributes, as model updates."""
    candidates: dict[str, Any] = {
        "description": detail.description,
        "page_count": detail.page_count,
        "series": [detail.series] if detail.series is not None else None,
        "authors": [RawAuthor(name=name) for name in detail.authors] if detail.authors else None,
        "isbns": [detail.isbn] if detail.isbn else None,
        "published": detail.published,
    }
    return {key: value for key, value in candidates.items() if value}


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
