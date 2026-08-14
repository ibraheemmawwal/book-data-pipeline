"""The v0.1 ingestion run: discover, resolve, canonicalise, load.

This is the seam where the pure stages meet the world. Everything it composes
is already tested in isolation; what it adds is the order, the run record and
the accounting.

Resolution is async and loading is synchronous, and they are deliberately not
interleaved: a batch of candidates is resolved, then loaded. Mixing them would
put network latency inside a database transaction, which is how one slow source
becomes a lock held for minutes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Connection, Engine

from pipeline.config import Settings
from pipeline.db import build_engine
from pipeline.discover import read_manifest, stream_candidates
from pipeline.extract.goodreads import GoodreadsExtractor
from pipeline.extract.resolver import CatalogueResolver, Resolution
from pipeline.load import CatalogueLoader, record_attempts, record_rejection
from pipeline.models.domain import CandidateBook, CleanBook, SourceName
from pipeline.models.events import BookEvent
from pipeline.observability.runs import finalise_run, record_source_skip, start_run
from pipeline.source_health import (
    SourceCoolingDownError,
    ensure_not_cooling_down,
    record_refusal,
)
from pipeline.transform import canonicalise, unify_identity

logger = structlog.get_logger(__name__)

RESOLVE_BATCH = 50


@dataclass
class IngestReport:
    """What one run did, for the operator and for ``ingestion_runs``."""

    candidates: int = 0
    resolved: int = 0
    unresolved: int = 0
    observations: int = 0
    rejected: int = 0
    books_inserted: int = 0
    books_updated: int = 0
    books_unchanged: int = 0
    attempts: int = 0
    # Set by the phase 2 path so the barrier knows which run to close.
    run_id: UUID | None = None

    @property
    def status(self) -> str:
        """Terminal run status.

        ``partial_success`` when some candidates resolved and some did not.
        With a hierarchy of fallible sources that is the common case, and
        rounding it to success or failure would throw away the one number an
        operator actually wants.
        """
        if not self.candidates or self.resolved == 0:
            return "failed"
        return "success" if self.unresolved == 0 else "partial_success"


def _batched(items: Iterable[CandidateBook], size: int) -> Iterator[list[CandidateBook]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def candidate_source(settings: Settings, limit: int | None) -> Iterator[CandidateBook]:
    """Candidates from a prepared manifest, else straight from the dump.

    A manifest is preferred because it is the deterministic artefact; reading
    the dump directly is the convenience path for a first run.
    """
    manifest = settings.discovery_manifest_path
    if manifest.exists():
        logger.info("ingest.using_manifest", path=str(manifest))
        candidates = read_manifest(manifest)
        return islice(candidates, limit) if limit else candidates

    if settings.openlibrary_dump_path is None:
        msg = (
            "no candidate manifest and no PIPELINE_OPENLIBRARY_DUMP_PATH; nothing to discover from"
        )
        raise FileNotFoundError(msg)

    logger.info("ingest.using_dump", path=str(settings.openlibrary_dump_path))
    return stream_candidates(
        Path(settings.openlibrary_dump_path),
        languages=settings.discovery_language_set(),
        max_candidates=limit or settings.discovery_max_candidates,
    )


async def _resolve_batch(
    resolver: CatalogueResolver, batch: list[CandidateBook], *, concurrency: int
) -> list[Resolution]:
    """Resolve a batch with a bounded number of candidates in flight.

    This was sequential, on the stated grounds that Goodreads permits one
    request at a time. That rule is real, but serialising every candidate is
    the wrong place to enforce it: it also serialises Open Library, which is
    89% of a run's wall clock and perfectly happy to have several requests
    outstanding as long as they arrive at its published rate.

    What makes this safe is that each source's rate limiter now lives on an
    extractor held for the whole run, so the published rate is enforced across
    candidates rather than incidentally by the loop; Goodreads keeps its own
    one-at-a-time gate inside the resolver.

    Results stay in candidate order, because the run's accounting reads them
    alongside the batch that produced them.
    """
    gate = asyncio.Semaphore(concurrency)

    async def resolve_one(candidate: CandidateBook) -> Resolution:
        async with gate:
            return await resolver.resolve(candidate)

    return list(await asyncio.gather(*(resolve_one(c) for c in batch)))


def account_for(
    connection: Connection,
    run_id: UUID,
    resolutions: list[Resolution],
    report: IngestReport,
) -> list[CleanBook]:
    """Record what one batch produced and return what is loadable.

    Attempts and rejections are written before the load rather than after: if
    the load fails, the record of *why* each source was used still has to exist.
    """
    clean: list[CleanBook] = []
    for resolution in resolutions:
        if resolution.resolved:
            report.resolved += 1
        else:
            report.unresolved += 1

        report.attempts += record_attempts(connection, run_id, resolution.attempts)

        for rejection in resolution.rejections:
            record_rejection(connection, run_id, rejection, stage="extract")
            report.rejected += 1

        for_candidate: list[CleanBook] = []
        for observation in resolution.observations:
            report.observations += 1
            result = canonicalise(observation)
            if isinstance(result, CleanBook):
                for_candidate.append(result)
            else:
                record_rejection(connection, run_id, result, stage="transform")
                report.rejected += 1

        # The resolver knows these describe one candidate; the load layer only
        # sees independent records and merges by identity. Without this, a book
        # two sources both resolved becomes two canonical rows.
        try:
            clean.extend(unify_identity(for_candidate))
        except ValueError:
            # Genuinely different books behind one candidate. Load them apart
            # rather than fusing a pair nothing could separate again.
            logger.warning("ingest.identity_conflict", candidate=resolution.candidate.candidate_key)
            clean.extend(for_candidate)

    return clean


def _goodreads_skip_reason(settings: Settings, engine: Engine) -> str | None:
    """Why Goodreads sits this run out, or ``None`` to use it.

    Configuration first, then what the source itself said last time.

    A cooldown skips the *source*, not the run. The other three sources have no
    quarrel with us, and letting one unofficial source decide whether the
    catalogue grows would hand it a veto it has not earned — ingestion resolves
    the same candidates through Open Library, Google Books and Gutendex without
    it. This is the difference from enrichment and contested resolution, where
    Goodreads is the entire point of the run and there is nothing to continue.
    """
    reason = settings.skip_reason(SourceName.GOODREADS)
    if reason is not None:
        return reason

    try:
        with engine.begin() as connection:
            ensure_not_cooling_down(
                connection,
                SourceName.GOODREADS,
                cooldown=timedelta(minutes=settings.goodreads_cooldown_minutes),
            )
    except SourceCoolingDownError as error:
        return str(error)
    return None


def run_ingestion(
    settings: Settings, *, limit: int | None = None, engine: Engine | None = None
) -> IngestReport:
    """Execute one full ingestion and return what it did.

    The run row is opened before any work and closed on every path, so a run
    that crashed is distinguishable from one that never started.
    """
    report = IngestReport()
    active = engine or build_engine(settings.database_url)

    # Constructed only when both gates allow it, so an unconfigured run never
    # builds a client it must not use.
    goodreads_skip = _goodreads_skip_reason(settings, active)
    goodreads = GoodreadsExtractor(settings) if goodreads_skip is None else None

    resolver = CatalogueResolver(settings, goodreads=goodreads)
    loader = CatalogueLoader()

    with active.begin() as connection:
        run_id = start_run(connection)
        if goodreads_skip is not None:
            record_source_skip(connection, run_id, SourceName.GOODREADS, goodreads_skip)

    try:
        for batch in _batched(candidate_source(settings, limit), RESOLVE_BATCH):
            report.candidates += len(batch)
            resolutions = asyncio.run(
                _resolve_batch(resolver, batch, concurrency=settings.resolution_concurrency)
            )

            with active.begin() as connection:
                clean = account_for(connection, run_id, resolutions, report)

            if clean:
                loaded = loader.load(active, clean, run_id=run_id)
                report.books_inserted += loaded.books_inserted
                report.books_updated += loaded.books_updated
                report.books_unchanged += loaded.books_unchanged

            logger.info(
                "ingest.batch_complete",
                candidates=report.candidates,
                resolved=report.resolved,
                books=report.books_inserted,
            )
    except Exception:
        with active.begin() as connection:
            finalise_run(connection, run_id, status="failed")
        raise

    with active.begin() as connection:
        if goodreads is not None and goodreads.circuit_open:
            record_refusal(connection, run_id, SourceName.GOODREADS, "circuit opened")
        finalise_run(
            connection,
            run_id,
            status=report.status,
            records_extracted=report.observations,
            records_loaded=report.books_inserted + report.books_updated,
            records_rejected=report.rejected,
        )
    return report


def run_resolution_to_sink(
    settings: Settings,
    sink: Any,
    *,
    limit: int | None = None,
    engine: Engine | None = None,
) -> IngestReport:
    """Phase 2's extract stage: resolve candidates and publish raw events.

    The same discovery and resolution as ``run_ingestion``, but the observations
    go onto a topic instead of into the catalogue. Canonicalisation and loading
    move to the consumers, which is what lets a slow load stop blocking
    ingestion rather than stalling the whole run behind it.

    Attempts and rejections are still written here. They belong to the
    resolution that produced them, and a consumer has no way to reconstruct why
    a source was skipped.
    """
    report = IngestReport()
    active = engine or build_engine(settings.database_url)

    goodreads_skip = _goodreads_skip_reason(settings, active)
    goodreads = GoodreadsExtractor(settings) if goodreads_skip is None else None
    resolver = CatalogueResolver(settings, goodreads=goodreads)

    with active.begin() as connection:
        run_id = start_run(connection)
        if goodreads_skip is not None:
            record_source_skip(connection, run_id, SourceName.GOODREADS, goodreads_skip)

    try:
        for batch in _batched(candidate_source(settings, limit), RESOLVE_BATCH):
            report.candidates += len(batch)
            resolutions = asyncio.run(
                _resolve_batch(resolver, batch, concurrency=settings.resolution_concurrency)
            )

            events: list[BookEvent] = []
            with active.begin() as connection:
                for resolution in resolutions:
                    if resolution.resolved:
                        report.resolved += 1
                    else:
                        report.unresolved += 1

                    report.attempts += record_attempts(connection, run_id, resolution.attempts)
                    for rejection in resolution.rejections:
                        record_rejection(connection, run_id, rejection, stage="extract")
                        report.rejected += 1

                    for observation in resolution.observations:
                        report.observations += 1
                        events.append(
                            BookEvent(
                                run_id=run_id,
                                source=observation.source,
                                source_id=observation.source_id,
                                payload=observation.raw_payload,
                            )
                        )

            if events:
                sink.emit(events)
                # Flushed per batch, so a crash costs one batch of re-resolved
                # candidates rather than the whole run's external calls.
                sink.flush()

            logger.info(
                "resolution.batch_produced",
                candidates=report.candidates,
                produced=report.observations,
            )
    except Exception:
        with active.begin() as connection:
            finalise_run(connection, run_id, status="failed")
        raise

    if goodreads is not None and goodreads.circuit_open:
        with active.begin() as connection:
            record_refusal(connection, run_id, SourceName.GOODREADS, "circuit opened")

    report.run_id = run_id
    return report
