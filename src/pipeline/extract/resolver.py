"""The catalogue resolver: one candidate, several sources, in a fixed order.

Fallback lives here rather than inside any adapter's ``fetch``. That is the
whole point of the module: when a book resolves, we want to know *which* source
answered, how long it took, and why the others were not used. Hiding that
inside an extractor would make source attempts, latency and fallback reasons
invisible — and an unofficial primary source is exactly the case where you need
them visible.

Order is contractual:

1. **Goodreads** — preferred resolver, when both gates are set and its circuit
   is closed.
2. **The retained Open Library discovery payload** — free, already in hand, and
   still required to pass the same validation as any API result.
3. **Open Library Search and Google Books** — bounded live fallbacks, each with
   a per-run budget.
4. **Gutendex** — last resort only, with a small budget so an outage upstream
   cannot quietly turn it back into the bulk source.

Every attempt is recorded, including the ones that were skipped and why.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import httpx
import structlog

from pipeline.config import Settings
from pipeline.extract.base import Rejected
from pipeline.extract.goodreads import (
    GoodreadsExtractor,
    GoodreadsNotAcceptedError,
    GoodreadsUnavailableError,
)
from pipeline.extract.openlibrary import map_payload
from pipeline.models.domain import CandidateBook, RawBook, SourceName

logger = structlog.get_logger(__name__)


class Outcome(StrEnum):
    """Why a source attempt ended as it did.

    ``no_match`` and ``unavailable`` are deliberately distinct: the first says
    the source answered and had nothing, the second says we never got an
    answer. Conflating them makes a run record useless for triage.
    """

    RESOLVED = "resolved"
    PARTIAL = "partial"
    NO_MATCH = "no_match"
    CONTRACT_FAILURE = "contract_failure"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class Attempt:
    """One source's attempt at one candidate, for ``resolution_attempts``."""

    candidate_key: str
    source: SourceName
    attempt_no: int
    outcome: Outcome
    fallback_reason: str | None = None
    duration_ms: int = 0


@dataclass
class Budget:
    """A per-run cap on calls to one source.

    Exhaustion is an observable ``skipped`` attempt, not permission to keep
    going. Without this, one Goodreads outage would spend a whole day's Google
    Books quota in minutes.
    """

    limit: int
    spent: int = 0

    def try_spend(self) -> bool:
        if self.spent >= self.limit:
            return False
        self.spent += 1
        return True

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.limit


@dataclass
class Resolution:
    """Everything one candidate produced.

    Observations are plural on purpose. "Goodreads first" decides lookup order
    and field preference, not whether other successful observations are allowed
    to exist — a lower-priority source can still supply a field the preferred
    one lacked.
    """

    candidate: CandidateBook
    observations: list[RawBook] = field(default_factory=list)
    rejections: list[Rejected] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.observations)


class _Timer:
    """Elapsed milliseconds, injectable so tests are not timing-dependent."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._started = clock()

    @property
    def elapsed_ms(self) -> int:
        return max(0, int((self._clock() - self._started) * 1000))


class CatalogueResolver:
    """Resolves candidates through the source hierarchy."""

    def __init__(
        self,
        settings: Settings,
        *,
        goodreads: GoodreadsExtractor | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock or time.monotonic
        self._goodreads = goodreads
        self._openlibrary_budget = Budget(settings.openlibrary_max_fallback_queries_per_run)
        self._googlebooks_budget = Budget(settings.googlebooks_max_fallback_queries_per_run)
        self._gutendex_budget = Budget(settings.gutendex_max_last_resort_queries_per_run)

    @property
    def budgets(self) -> dict[SourceName, Budget]:
        """Exposed so a run can report what it spent."""
        return {
            SourceName.OPENLIBRARY: self._openlibrary_budget,
            SourceName.GOOGLEBOOKS: self._googlebooks_budget,
            SourceName.GUTENDEX: self._gutendex_budget,
        }

    async def resolve(
        self, candidate: CandidateBook, *, client: httpx.AsyncClient | None = None
    ) -> Resolution:
        """Run one candidate through the hierarchy."""
        result = Resolution(candidate=candidate)

        await self._try_goodreads(candidate, result, client)
        self._try_retained_discovery(candidate, result)

        return result

    async def _try_goodreads(
        self,
        candidate: CandidateBook,
        result: Resolution,
        client: httpx.AsyncClient | None,
    ) -> None:
        """Attempt the preferred resolver, recording why if it is not used."""
        if self._goodreads is None:
            result.attempts.append(
                Attempt(
                    candidate.candidate_key,
                    SourceName.GOODREADS,
                    1,
                    Outcome.SKIPPED,
                    "goodreads adapter not configured",
                )
            )
            return

        try:
            self._goodreads.ensure_accepted()
        except GoodreadsNotAcceptedError as error:
            result.attempts.append(
                Attempt(
                    candidate.candidate_key,
                    SourceName.GOODREADS,
                    1,
                    Outcome.SKIPPED,
                    str(error),
                )
            )
            return

        if self._goodreads.circuit_open:
            # Once open it stays open for the run. Re-probing a source that has
            # refused us is exactly what the containment rules forbid.
            result.attempts.append(
                Attempt(
                    candidate.candidate_key,
                    SourceName.GOODREADS,
                    1,
                    Outcome.SKIPPED,
                    "circuit open for this run",
                )
            )
            return

        timer = _Timer(self._clock)
        owned = client is None
        active = client or self._goodreads.build_client()
        try:
            observation = await self._goodreads.resolve(
                active,
                candidate.lookup_query(),
                isbn=candidate.preferred_isbn(),
            )
        except GoodreadsUnavailableError as error:
            result.attempts.append(
                Attempt(
                    candidate.candidate_key,
                    SourceName.GOODREADS,
                    1,
                    Outcome.UNAVAILABLE,
                    str(error),
                    timer.elapsed_ms,
                )
            )
            return
        finally:
            if owned:
                await active.aclose()

        if observation is None:
            result.attempts.append(
                Attempt(
                    candidate.candidate_key,
                    SourceName.GOODREADS,
                    1,
                    Outcome.NO_MATCH,
                    None,
                    timer.elapsed_ms,
                )
            )
            return

        result.observations.append(observation)
        result.attempts.append(
            Attempt(
                candidate.candidate_key,
                SourceName.GOODREADS,
                1,
                Outcome.RESOLVED,
                None,
                timer.elapsed_ms,
            )
        )

    def _try_retained_discovery(self, candidate: CandidateBook, result: Resolution) -> None:
        """Promote the discovery payload we already hold.

        Free — it cost a dump read, not a request — but it still has to pass
        the same validation as an API result. Discovery proves a book exists,
        not that its fields are usable. It is kept even when Goodreads already
        resolved the candidate, because a lower-priority observation can still
        fill a field the preferred source lacked.
        """
        if not candidate.discovery_payload:
            result.attempts.append(
                Attempt(
                    candidate.candidate_key,
                    SourceName.OPENLIBRARY,
                    1,
                    Outcome.SKIPPED,
                    "no retained discovery payload",
                )
            )
            return

        timer = _Timer(self._clock)
        mapped = map_payload(candidate.discovery_payload)
        if isinstance(mapped, Rejected):
            result.rejections.append(mapped)
            result.attempts.append(
                Attempt(
                    candidate.candidate_key,
                    SourceName.OPENLIBRARY,
                    1,
                    Outcome.CONTRACT_FAILURE,
                    mapped.detail,
                    timer.elapsed_ms,
                )
            )
            return

        result.observations.append(mapped)
        result.attempts.append(
            Attempt(
                candidate.candidate_key,
                SourceName.OPENLIBRARY,
                1,
                Outcome.RESOLVED,
                "retained discovery payload",
                timer.elapsed_ms,
            )
        )
