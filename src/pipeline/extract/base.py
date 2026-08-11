"""Shared extraction machinery.

Three concerns live here because all three extractors need them and getting any
of them wrong is a data-quality or a good-citizenship failure rather than a
style preference.

**Retry policy.** Retrying a 404 wastes a source's goodwill and cannot succeed;
not retrying a 429 gets the pipeline blocked. So the policy is explicit about
which failures can plausibly recover, and a source that never recovers raises a
typed terminal failure the Airflow barrier can record rather than an ``httpx``
internal that reads as a crash.

**Rate limiting.** A token bucket, held across an extractor's whole run rather
than per request, so concurrency cannot slip two calls through one token.

**Per-item isolation.** One malformed record must become a recorded rejection,
never an aborted page. Extractors therefore yield a union: a validated
``RawBook`` or a ``Rejected`` carrying enough context to debug it later without
the original response.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pipeline.models.domain import RawBook, SourceName

# Status codes worth trying again. Everything else in 4xx is a request we got
# wrong, and repeating it verbatim will keep being wrong.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

HTTP_ERROR_THRESHOLD = 400
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY_SECONDS = 1.0
DEFAULT_MAX_DELAY_SECONDS = 30.0

Sleep = Callable[[float], Awaitable[None]]
BeforeAttempt = Callable[[], Awaitable[None]]


class SourceUnavailableError(Exception):
    """A source failed in a way retrying will not fix.

    Carries the last status code so ``source_runs.error`` records something
    more useful than a stack trace, and so the barrier can distinguish "the
    source is down" from "our request was malformed".
    """

    def __init__(self, source: str, message: str, *, status_code: int | None = None) -> None:
        self.source = source
        self.status_code = status_code
        super().__init__(f"{source}: {message}")


class InvalidSourceRecordError(ValueError):
    """A provider returned JSON with an unexpected per-record shape."""


def require_object(value: object, location: str) -> dict[str, Any]:
    """Return a JSON object or raise a rejection-friendly shape error."""
    if not isinstance(value, dict):
        msg = f"{location} must be an object, got {type(value).__name__}"
        raise InvalidSourceRecordError(msg)
    return value


def optional_list(document: dict[str, Any], field: str) -> list[Any]:
    """Read an optional JSON array without treating strings as iterables."""
    value = document.get(field)
    if value is None:
        return []
    if not isinstance(value, list):
        msg = f"{field} must be an array, got {type(value).__name__}"
        raise InvalidSourceRecordError(msg)
    return value


def optional_object(document: dict[str, Any], field: str) -> dict[str, Any]:
    """Read an optional JSON object."""
    value = document.get(field)
    if value is None:
        return {}
    return require_object(value, field)


def string_list(document: dict[str, Any], field: str) -> list[str]:
    """Read an optional array whose entries must all be strings."""
    values = optional_list(document, field)
    if any(not isinstance(value, str) for value in values):
        msg = f"{field} must contain only strings"
        raise InvalidSourceRecordError(msg)
    return values


def record_error_detail(error: InvalidSourceRecordError | ValidationError) -> str:
    """Render a bounded rejection reason for storage and logs."""
    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors()
        )[:500]
    return str(error)[:500]


class ExtractionRequest(BaseModel):
    """What one extractor run should fetch.

    Deliberately has no ``since``. The three sources expose no uniform
    incremental cursor — Gutendex publishes no modification timestamp at all —
    so a ``since`` parameter would imply an incremental guarantee none of them
    can honour. Repeat ingestion is made safe by source identity and canonical
    idempotency instead.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_records: int = Field(gt=0)
    query: str | None = None


@dataclass(frozen=True, slots=True)
class Rejected:
    """A source record that failed validation.

    Kept rather than dropped: a pipeline that silently discards bad rows is one
    nobody can trust, and ``rejected_records`` is where these land.
    """

    source: SourceName
    source_id: str | None
    raw_payload: Any
    rejection_code: str
    detail: str | None


ExtractedItem = RawBook | Rejected


@runtime_checkable
class Extractor(Protocol):
    """What every source implements."""

    source_name: SourceName

    def fetch(self, request: ExtractionRequest) -> AsyncIterator[ExtractedItem]:
        """Yield validated records and rejections until the limit is reached.

        Raises:
            SourceUnavailableError: the source failed terminally. Individual bad
                records never raise; they are yielded as ``Rejected``.
        """
        ...


def build_client(
    *,
    user_agent: str,
    connect_timeout: float = 5.0,
    read_timeout: float = 30.0,
) -> httpx.AsyncClient:
    """An HTTP client with a timeout on every phase.

    ``httpx`` defaults are generous and a missing write or pool timeout is how
    a hung socket turns into a permanently stuck Airflow task, so all four are
    set explicitly.
    """
    return httpx.AsyncClient(
        headers={"User-Agent": user_agent, "Accept": "application/json"},
        timeout=httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=read_timeout,
            pool=connect_timeout,
        ),
        follow_redirects=True,
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse ``Retry-After``, ignoring anything we cannot read.

    Only the delta-seconds form is handled. The HTTP-date form is rare on these
    APIs, and a malformed header must degrade to ordinary backoff rather than
    crash a run.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _backoff(attempt: int, base_delay: float, max_delay: float) -> float:
    """Exponential ceiling with full jitter.

    Jitter matters more than the exponent: without it every client that hit the
    same outage retries in lockstep and rebuilds the spike that caused it.
    """
    ceiling = min(base_delay * (2**attempt), max_delay)
    return random.uniform(0, ceiling) if ceiling > 0 else 0.0


async def request_with_retries(  # noqa: PLR0913
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay: float = DEFAULT_MAX_DELAY_SECONDS,
    sleep: Sleep | None = None,
    before_attempt: BeforeAttempt | None = None,
    source: str = "http",
) -> httpx.Response:
    """Issue a request, retrying only what can plausibly recover.

    Raises:
        SourceUnavailableError: attempts were exhausted, or the response was a
            client error that retrying cannot fix.
    """
    wait = sleep if sleep is not None else asyncio.sleep
    last_status: int | None = None
    last_detail = "no attempt was made"

    for attempt in range(max_attempts):
        if before_attempt is not None:
            await before_attempt()
        try:
            response = await client.request(method, url, params=params)
        except httpx.TransportError as error:
            # Connection reset, DNS blip, read timeout: all plausibly transient.
            last_detail = f"{type(error).__name__}: {error}"
        else:
            if response.status_code < HTTP_ERROR_THRESHOLD:
                return response

            last_status = response.status_code
            last_detail = f"HTTP {response.status_code}"

            if response.status_code not in RETRYABLE_STATUS:
                raise SourceUnavailableError(
                    source,
                    f"{last_detail} is not retryable",
                    status_code=response.status_code,
                )

            requested = _retry_after_seconds(response)
            if requested is not None:
                # Honour the source's own answer, but never let it hold an
                # Airflow task open for longer than the run can afford.
                if attempt < max_attempts - 1:
                    await wait(min(requested, max_delay))
                continue

        if attempt < max_attempts - 1:
            await wait(_backoff(attempt, base_delay, max_delay))

    raise SourceUnavailableError(
        source,
        f"exhausted {max_attempts} attempts ({last_detail})",
        status_code=last_status,
    )


class TokenBucket:
    """Serialises calls to at most ``rate_per_second``.

    Held for an extractor's whole run rather than per request, and guarded by a
    lock, because two coroutines that read the clock at the same moment would
    otherwise both decide it was their turn.
    """

    def __init__(
        self,
        rate_per_second: float,
        *,
        sleep: Sleep | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if rate_per_second <= 0:
            msg = "rate_per_second must be positive"
            raise ValueError(msg)
        self._interval = 1.0 / rate_per_second
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._monotonic = monotonic if monotonic is not None else time.monotonic
        self._lock = asyncio.Lock()
        self._next_allowed: float | None = None

    async def acquire(self) -> None:
        """Block until another request is permitted."""
        async with self._lock:
            now = self._monotonic()
            if self._next_allowed is not None and now < self._next_allowed:
                await self._sleep(self._next_allowed - now)
                # Re-read the clock rather than assuming the sleep landed
                # exactly on the slot. Scheduling from the intended slot instead
                # compounds whenever a sleep returns early: each acquire books a
                # slot further into the future than the last, and the waits grow
                # without bound. Reading the clock is self-correcting, and the
                # contract is "at most this rate", so drifting slightly slow is
                # the safe direction to be wrong in.
                now = self._monotonic()
            self._next_allowed = now + self._interval
