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
from itertools import islice
from pathlib import Path
from uuid import UUID

import structlog
from sqlalchemy import Connection, Engine, create_engine

from pipeline.config import Settings
from pipeline.discover import read_manifest, stream_candidates
from pipeline.extract.goodreads import GoodreadsExtractor
from pipeline.extract.resolver import CatalogueResolver, Resolution
from pipeline.load import CatalogueLoader, record_attempts, record_rejection
from pipeline.models.domain import CandidateBook, CleanBook, SourceName
from pipeline.observability.runs import finalise_run, record_source_skip, start_run
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
    resolver: CatalogueResolver, batch: list[CandidateBook]
) -> list[Resolution]:
    """Resolve a batch one candidate at a time.

    Sequential on purpose: Goodreads permits one request in flight, and
    resolving concurrently would breach that from the very first candidate.
    """
    return [await resolver.resolve(candidate) for candidate in batch]


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


def run_ingestion(
    settings: Settings, *, limit: int | None = None, engine: Engine | None = None
) -> IngestReport:
    """Execute one full ingestion and return what it did.

    The run row is opened before any work and closed on every path, so a run
    that crashed is distinguishable from one that never started.
    """
    report = IngestReport()
    active = engine or create_engine(settings.database_url)

    # Constructed only when both gates allow it, so an unconfigured run never
    # builds a client it must not use.
    goodreads_skip = settings.skip_reason(SourceName.GOODREADS)
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
            resolutions = asyncio.run(_resolve_batch(resolver, batch))

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
        finalise_run(
            connection,
            run_id,
            status=report.status,
            records_extracted=report.observations,
            records_loaded=report.books_inserted + report.books_updated,
            records_rejected=report.rejected,
        )
    return report
