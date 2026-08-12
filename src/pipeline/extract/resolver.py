"""The catalogue resolver: one candidate, several sources, in a fixed order.

Fallback lives here rather than inside any adapter's ``fetch``. That is the
whole point of the module: when a book resolves, we want to know *which* source
answered, how long it took, and why the others were not used. Hiding that
inside an extractor would make source attempts, latency and fallback reasons
invisible — and an unofficial primary source is exactly the case where you need
them visible.

Order is contractual, and documented sources carry it:

1. **The retained Open Library discovery payload** — free, already in hand, and
   still required to pass the same validation as any API result.
2. **Open Library Search and Google Books** — bounded live lookups, each with a
   per-run budget.
3. **Gutendex** — public-domain works, bounded, so an outage upstream cannot
   quietly turn it into the bulk source.

**Goodreads** sits outside that order. The adapter is wired and the code path
exists — ``goodreads_in_resolution`` turns it on, and the contested-resolution
flow uses it directly — but ingestion does not consult it by default. Asking it
about every book is what a preferred-resolver position means, and that is what
its terms do not support; adjudicating the minority where documented sources
conflict is a different claim.

Every attempt is recorded, including the ones that were skipped and why.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import httpx
import structlog

from pipeline.config import Settings
from pipeline.extract import build_extractor
from pipeline.extract.base import (
    ExtractionRequest,
    Extractor,
    QuotaExhaustedError,
    Rejected,
    SourceUnavailableError,
)
from pipeline.extract.goodreads import (
    GoodreadsExtractor,
    GoodreadsNotAcceptedError,
    GoodreadsUnavailableError,
)
from pipeline.extract.googlebooks import MissingCredentialError
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

    def spend_all(self) -> None:
        """Retire this budget for the rest of the run."""
        self.spent = self.limit

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
        # One extractor per source for the whole run, because that is the only
        # scope at which a rate limiter means anything. Rebuilding it per
        # attempt gave every call a fresh TokenBucket with nothing to remember,
        # so the limit was never enforced — sequential resolution was the only
        # thing keeping this polite, which made politeness a property of the
        # loop shape rather than of the code that claims to provide it.
        self._extractors: dict[SourceName, Extractor] = {}
        # Goodreads permits one request in flight. That is a per-source rule,
        # not a reason to resolve candidates one at a time, so it is held here
        # rather than by serialising the whole run.
        # Rebound per loop for the same reason as the rate limiters: a run
        # opens one event loop per batch, and a primitive created here would
        # belong to whichever loop first awaited it.
        self._goodreads_gate: asyncio.Semaphore | None = None
        self._goodreads_gate_loop: asyncio.AbstractEventLoop | None = None
        self._openlibrary_budget = Budget(settings.openlibrary_max_fallback_queries_per_run)
        self._googlebooks_budget = Budget(settings.googlebooks_max_fallback_queries_per_run)
        self._gutendex_budget = Budget(settings.gutendex_max_last_resort_queries_per_run)

    def _goodreads_gate_for_this_loop(self) -> asyncio.Semaphore:
        """The one-request-in-flight gate, belonging to the running loop."""
        loop = asyncio.get_running_loop()
        if self._goodreads_gate is None or self._goodreads_gate_loop is not loop:
            self._goodreads_gate = asyncio.Semaphore(1)
            self._goodreads_gate_loop = loop
        return self._goodreads_gate

    def _extractor_for(self, source: SourceName) -> Extractor:
        """The run's extractor for ``source``, built once and kept.

        Each fetch still opens its own HTTP client; what is shared is the rate
        limiter, which is the part that has to outlive a single call.
        """
        extractor = self._extractors.get(source)
        if extractor is None:
            extractor = build_extractor(source, self._settings)
            self._extractors[source] = extractor
        return extractor

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

        # Documented sources carry ingestion. Goodreads is wired and ready —
        # contested resolution uses it, and search may later — but it is not on
        # the path every candidate takes, because a preferred-resolver position
        # means asking it about every book, and that is the thing its terms do
        # not support.
        if self._settings.goodreads_in_resolution:
            # One request in flight, whatever else the run is doing. The gate
            # is here rather than around the whole candidate so the documented
            # sources still overlap.
            async with self._goodreads_gate_for_this_loop():
                await self._try_goodreads(candidate, result, client)

        self._try_retained_discovery(candidate, result)

        # Two modes, and the difference is what the run is for.
        #
        # Filling gaps (the default): skip the documented APIs once anything
        # valid exists, so a working run does not quietly spend its whole
        # fallback budget re-answering a question already answered.
        #
        # Enriching: query them anyway. A book resolved by exactly one source
        # has no cross-source provenance and cannot disagree with anything, so
        # a catalogue built purely by gap-filling can never answer "do the
        # sources agree" — the question this pipeline exists to make possible.
        # Budgets still bound it.
        if self._settings.enrich_from_documented_sources or not result.resolved:
            await self._try_live_fallbacks(candidate, result)

        # Gutendex, when nothing else produced a record — or, under
        # enrichment, as a third opinion.
        #
        # It only holds public-domain works, so enriching with it reaches a
        # subset of the catalogue and leaves the rest untouched. That is the
        # point: on a nineteenth-century classic it is a genuinely independent
        # reading of the same book, and three sources disagreeing is a stronger
        # signal than two.
        #
        # The budget still bounds it, and that guard is what keeps this from
        # promoting a public-domain mirror back into being the bulk source —
        # the failure the small budget exists to prevent.
        if self._settings.enrich_from_documented_sources or not result.resolved:
            await self._try_gutendex(candidate, result)

        return result

    async def _try_live_fallbacks(self, candidate: CandidateBook, result: Resolution) -> None:
        """Open Library Search and Google Books, each under its own budget."""
        for source, budget in (
            (SourceName.OPENLIBRARY, self._openlibrary_budget),
            (SourceName.GOOGLEBOOKS, self._googlebooks_budget),
        ):
            await self._try_bulk_source(candidate, result, source, budget, attempt_no=2)

    async def _try_gutendex(self, candidate: CandidateBook, result: Resolution) -> None:
        await self._try_bulk_source(
            candidate,
            result,
            SourceName.GUTENDEX,
            self._gutendex_budget,
            attempt_no=1,
        )

    async def _try_bulk_source(
        self,
        candidate: CandidateBook,
        result: Resolution,
        source: SourceName,
        budget: Budget,
        *,
        attempt_no: int,
    ) -> None:
        """One bounded lookup against a documented API.

        Every exit records an attempt. A source that was configured off, out of
        budget, or simply had no match must all be distinguishable afterwards —
        otherwise a run with a silent hole looks like a run with nothing to
        find.
        """
        skip_reason = self._settings.skip_reason(source)
        if skip_reason is not None:
            result.attempts.append(
                Attempt(
                    candidate.candidate_key,
                    source,
                    attempt_no,
                    Outcome.SKIPPED,
                    skip_reason,
                )
            )
            return

        if not budget.try_spend():
            result.attempts.append(
                Attempt(
                    candidate.candidate_key,
                    source,
                    attempt_no,
                    Outcome.SKIPPED,
                    f"per-run budget of {budget.limit} exhausted",
                )
            )
            return

        timer = _Timer(self._clock)
        extractor = self._extractor_for(source)
        request = ExtractionRequest(max_records=1, query=candidate.lookup_query())

        try:
            async for item in extractor.fetch(request):
                if isinstance(item, Rejected):
                    result.rejections.append(item)
                    continue
                result.observations.append(item)
                result.attempts.append(
                    Attempt(
                        candidate.candidate_key,
                        source,
                        attempt_no,
                        Outcome.RESOLVED,
                        None,
                        timer.elapsed_ms,
                    )
                )
                return
        except QuotaExhaustedError as error:
            # The allowance is spent for the day, so the next candidate would
            # buy the same answer at the price of another request. Burn the
            # budget instead: every later candidate then skips this source with
            # a reason, and the remaining allowance survives for the next run.
            budget.spend_all()
            result.attempts.append(
                Attempt(
                    candidate.candidate_key,
                    source,
                    attempt_no,
                    Outcome.UNAVAILABLE,
                    str(error),
                    timer.elapsed_ms,
                )
            )
            return
        except SourceUnavailableError as error:
            result.attempts.append(
                Attempt(
                    candidate.candidate_key,
                    source,
                    attempt_no,
                    Outcome.UNAVAILABLE,
                    str(error),
                    timer.elapsed_ms,
                )
            )
            return
        except MissingCredentialError as error:
            # Our configuration gap, not the provider's outage. Recording it as
            # unavailable would send someone hunting a fault at Google.
            result.attempts.append(
                Attempt(
                    candidate.candidate_key,
                    source,
                    attempt_no,
                    Outcome.SKIPPED,
                    str(error),
                    timer.elapsed_ms,
                )
            )
            return

        result.attempts.append(
            Attempt(
                candidate.candidate_key,
                source,
                attempt_no,
                Outcome.NO_MATCH,
                None,
                timer.elapsed_ms,
            )
        )

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
