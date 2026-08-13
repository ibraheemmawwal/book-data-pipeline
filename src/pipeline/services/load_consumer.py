"""The load consumer: books.clean in, catalogue out.

The last stage, and the one where redelivery has to be harmless. It is: the
load layer keys on ``(source, source_id)`` and compares a content hash, so
processing the same event twice changes no rows and moves no timestamp.

The offset is committed only after the database transaction has, which means a
crash between the two replays the record. That is the safe direction. The
unsafe direction — committing first — loses the book with nothing to show for
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from sqlalchemy import Engine

from pipeline.extract import map_payload
from pipeline.extract.base import Rejected
from pipeline.load import CatalogueLoader
from pipeline.messaging.contracts import Sink, Source
from pipeline.models.domain import CleanBook
from pipeline.models.events import BookEvent, PartitionMarker
from pipeline.observability.markers import finalise_if_complete, record_marker, runs_awaiting
from pipeline.services.transform_consumer import dlq_event
from pipeline.transform import canonicalise

logger = structlog.get_logger(__name__)

# Records written per transaction. Each book costs about twelve statements, so
# the transaction itself is only two round trips of fourteen — batching saves
# that overhead, not the statements. Kept small because everything in a batch
# is redelivered together if the process dies mid-write, and because a batch is
# also the longest a record can sit unwritten.
DEFAULT_LOAD_BATCH = 50


@dataclass
class LoadStats:
    """What one service run did."""

    loaded: int = 0
    rejected: int = 0
    markers: int = 0
    runs_finalised: int = 0
    errors: list[str] = field(default_factory=list)


class LoadConsumer:
    """Writes clean events into the catalogue and closes runs."""

    def __init__(
        self,
        engine: Engine,
        source: Source[Any],
        dlq_sink: Sink[Any],
        *,
        clean_topic: str = "books.clean",
        batch_size: int = DEFAULT_LOAD_BATCH,
    ) -> None:
        self._engine = engine
        self._source = source
        self._dlq = dlq_sink
        self._clean_topic = clean_topic
        self._loader = CatalogueLoader()
        self._batch_size = max(1, batch_size)

    def sweep(self, stats: LoadStats) -> None:
        """Close any run whose markers all arrived before a previous restart."""
        with self._engine.begin() as connection:
            pending = runs_awaiting(connection, self._clean_topic)

        for run_id in pending:
            with self._engine.begin() as connection:
                if finalise_if_complete(connection, run_id, self._clean_topic):
                    stats.runs_finalised += 1

    def run(self, stats: LoadStats | None = None) -> LoadStats:
        """Consume until the source stops, writing in batches.

        Records accumulate and are written together, because one transaction
        per book spends two of its fourteen round trips on BEGIN and COMMIT and
        the database is not local.

        Two orderings are load-bearing. A batch is always written before the
        marker that follows it, because the marker is what declares a run
        complete and a run must not be complete while its records are still in
        this list. And the offset is committed only after the write, so a crash
        replays the batch rather than losing it — which the idempotent load
        makes harmless, and which is the whole reason batching is safe here.
        """
        result = stats or LoadStats()
        self.sweep(result)

        batch: list[CleanBook] = []
        batch_run: Any = None

        for event in self._source.consume():
            if isinstance(event, PartitionMarker):
                batch = self._flush(batch, batch_run, result)
                self._handle_marker(event, result)
                self._source.commit()
                continue

            prepared = self._prepare(event, result)
            if prepared is None:
                # Parked. Everything ahead of it must be written before the
                # offset moves past it.
                batch = self._flush(batch, batch_run, result)
                self._source.commit()
                continue

            # load() takes one run_id for the whole call, so a change of run is
            # a batch boundary. Runs are sequential per partition, so this is
            # rare rather than a per-record cost.
            if batch and event.run_id != batch_run:
                batch = self._flush(batch, batch_run, result)
            batch_run = event.run_id
            batch.append(prepared)

            if len(batch) >= self._batch_size:
                batch = self._flush(batch, batch_run, result)
                self._source.commit()

        if batch:
            self._flush(batch, batch_run, result)
            self._source.commit()
        return result

    def _flush(self, batch: list[CleanBook], run_id: Any, stats: LoadStats) -> list[CleanBook]:
        """Write a batch and return an empty one."""
        if batch:
            outcome = self._loader.load(self._engine, batch, run_id=run_id)
            stats.loaded += outcome.records_loaded
        return []

    def _prepare(self, event: BookEvent, stats: LoadStats) -> CleanBook | None:
        """Map and validate one event, parking it if it cannot be loaded."""
        mapped = map_payload(event.source, event.payload)
        if isinstance(mapped, Rejected):
            self._park(event, "invalid_record", mapped.detail or "", stats)
            return None

        cleaned = canonicalise(mapped)
        if not isinstance(cleaned, CleanBook):
            self._park(event, cleaned.rejection_code, cleaned.detail or "", stats)
            return None
        return cleaned

    def _park(self, event: BookEvent, code: str, detail: str, stats: LoadStats) -> None:
        """Park a record the loader cannot use.

        Flushed before returning, so a DLQ failure raises and the offset is
        never committed — the record is redelivered rather than lost.
        """
        self._dlq.emit([dlq_event(event, code, detail, 1)])
        self._dlq.flush()
        stats.rejected += 1
        logger.warning("load.parked", code=code, source_id=event.source_id)

    def _handle_marker(self, marker: PartitionMarker, stats: LoadStats) -> None:
        """Record a clean boundary marker and finalise the run once complete."""
        with self._engine.begin() as connection:
            record_marker(connection, marker.run_id, marker.topic, marker.partition)
            finalised = finalise_if_complete(connection, marker.run_id, marker.topic)

        stats.markers += 1
        if finalised:
            stats.runs_finalised += 1
