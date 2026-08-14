"""Completing Goodreads records that arrived without their detail.

An export gives a title, its authors, a rating and a cover. It gives no year,
no ISBN, no page count and no series, because those live on the book's own
page. Ten thousand such records are a backlog of fetches, not a fault, and this
is the thing that works through them.

It is deliberately not part of ingestion. Ingestion resolves candidates it has
never seen; this revisits records already held, at a pace set by what the
source will bear. Running them together would put one run's failure in charge
of the other's budget, and it is the enrichment that gets blocked.

The same double gate as every other Goodreads path applies, for the same
reason: the source's terms restrict automated collection, and a bulk backlog is
exactly where that deserves a deliberate acknowledgement rather than a default.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Engine, text

from pipeline.config import Settings
from pipeline.db import build_engine
from pipeline.extract import map_payload
from pipeline.extract.base import Rejected
from pipeline.extract.goodreads import GoodreadsExtractor
from pipeline.load import CatalogueLoader
from pipeline.models.domain import CleanBook, SourceName
from pipeline.observability.runs import finalise_run, start_run
from pipeline.source_health import ensure_not_cooling_down, record_refusal
from pipeline.transform import canonicalise

logger = structlog.get_logger(__name__)

# Records that have never been fetched: an export supplied them, so they carry
# neither the book page nor the editions page.
UNENRICHED = """
    SELECT book_id, source_id, raw_payload
    FROM book_sources
    WHERE source = 'goodreads'
      AND NOT (raw_payload ? '_edition')
      AND NOT (raw_payload ? '_detail')
    ORDER BY last_seen_at
    LIMIT :limit
"""


@dataclass
class EnrichReport:
    """What one enrichment pass did."""

    pending: int = 0
    queried: int = 0
    enriched: int = 0
    unchanged: int = 0
    failed: int = 0
    loaded: int = 0
    run_id: UUID | None = None
    refused: bool = False
    errors: list[str] = field(default_factory=list)


def find_unenriched(engine: Engine, *, limit: int) -> list[dict[str, Any]]:
    """Goodreads records still missing their detail pages.

    Oldest first, by when the record was last seen. A run that took the newest
    would revisit the same head of the queue every time and never reach the
    books imported first.
    """
    with engine.begin() as connection:
        rows = connection.execute(text(UNENRICHED), {"limit": limit}).all()
    return [
        {"book_id": row.book_id, "source_id": row.source_id, "payload": row.raw_payload}
        for row in rows
    ]


def count_unenriched(engine: Engine) -> int:
    """How many records are waiting, for a report that means something."""
    with engine.begin() as connection:
        return int(
            connection.execute(
                text(
                    "SELECT count(*) FROM book_sources WHERE source='goodreads' "
                    "AND NOT (raw_payload ? '_edition') AND NOT (raw_payload ? '_detail')"
                )
            ).scalar_one()
        )


async def _enrich_all(
    extractor: GoodreadsExtractor,
    engine: Engine,
    records: list[dict[str, Any]],
    report: EnrichReport,
) -> None:
    loader = CatalogueLoader()
    client = extractor.build_client()

    try:
        for record in records:
            if extractor.circuit_open:
                # Stop for the run either way — the backlog is not going
                # anywhere. Only a refusal earns the cross-run cooldown: a run
                # ended by upstream 5xx should let the next one try.
                report.refused = extractor.refused
                report.errors.append(f"circuit opened: {extractor.circuit_reason}")
                break

            observation = map_payload(SourceName.GOODREADS, record["payload"])
            if isinstance(observation, Rejected):
                report.failed += 1
                continue

            report.queried += 1
            full = await extractor.enrich_by_id(client, observation)
            if full is None:
                report.unchanged += 1
                continue

            cleaned = canonicalise(full)
            if not isinstance(cleaned, CleanBook):
                report.failed += 1
                report.errors.append(f"{record['source_id']}: {cleaned.detail}"[:120])
                continue

            report.enriched += 1
            outcome = loader.load(engine, [cleaned], run_id=report.run_id)
            report.loaded += outcome.records_loaded
    finally:
        await client.aclose()


def enrich_goodreads(
    settings: Settings,
    *,
    limit: int = 200,
    engine: Engine | None = None,
) -> EnrichReport:
    """Fetch detail for records that arrived without it.

    The observation keeps its own ``(source, source_id)``, so the loader
    attaches it to the book it already belongs to. Nothing is re-keyed and no
    duplicate can appear — which is the difference between this and the
    contested flow, where an answer arrives with an identity of its own.

    Raises:
        GoodreadsNotAcceptedError: either gate is unset.
        SourceCoolingDownError: Goodreads refused a recent run.
    """
    active = engine or build_engine(settings.database_url)
    report = EnrichReport()

    # Before anything observable, so a refused run leaves no run row behind
    # suggesting work was attempted.
    extractor = GoodreadsExtractor(settings)
    extractor.ensure_accepted()

    report.pending = count_unenriched(active)
    records = find_unenriched(active, limit=limit)
    if not records:
        logger.info("enrich.nothing_pending")
        return report

    # In the transaction that opens the run, and only once there is work to do.
    # Checking earlier would cost a query on every empty run; checking later
    # would leave a run row for work that never happened, and raising here
    # rolls the whole thing back.
    with active.begin() as connection:
        ensure_not_cooling_down(
            connection,
            SourceName.GOODREADS,
            cooldown=timedelta(minutes=settings.goodreads_cooldown_minutes),
        )
        report.run_id = start_run(connection)

    try:
        asyncio.run(_enrich_all(extractor, active, records, report))
    except Exception:
        with active.begin() as connection:
            finalise_run(connection, report.run_id, status="failed")
        raise

    with active.begin() as connection:
        if report.refused:
            # Written before the run is closed, so a crash between the two
            # cannot lose the one fact the next run needs.
            record_refusal(connection, report.run_id, SourceName.GOODREADS, "circuit opened")
        finalise_run(
            connection,
            report.run_id,
            status="success" if report.enriched else "partial_success",
            records_extracted=report.enriched,
            records_loaded=report.loaded,
        )

    logger.info(
        "enrich.complete",
        pending=report.pending,
        queried=report.queried,
        enriched=report.enriched,
        loaded=report.loaded,
    )
    return report
