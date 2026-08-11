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
    ) -> None:
        self._engine = engine
        self._source = source
        self._dlq = dlq_sink
        self._clean_topic = clean_topic
        self._loader = CatalogueLoader()

    def sweep(self, stats: LoadStats) -> None:
        """Close any run whose markers all arrived before a previous restart."""
        with self._engine.begin() as connection:
            pending = runs_awaiting(connection, self._clean_topic)

        for run_id in pending:
            with self._engine.begin() as connection:
                if finalise_if_complete(connection, run_id, self._clean_topic):
                    stats.runs_finalised += 1

    def run(self, stats: LoadStats | None = None) -> LoadStats:
        """Consume until the source stops."""
        result = stats or LoadStats()
        self.sweep(result)

        for event in self._source.consume():
            if isinstance(event, PartitionMarker):
                self._handle_marker(event, result)
            else:
                self._handle_book(event, result)

            # After the effect, never before. A crash between the two replays
            # the record, which the idempotent load makes harmless; committing
            # first would lose it with nothing to show for it.
            self._source.commit()

        return result

    def _handle_book(self, event: BookEvent, stats: LoadStats) -> None:
        """Load one record. Redelivery is a no-op by construction."""
        mapped = map_payload(event.source, event.payload)
        if isinstance(mapped, Rejected):
            self._park(event, "invalid_record", mapped.detail or "", stats)
            return

        cleaned = canonicalise(mapped)
        if not isinstance(cleaned, CleanBook):
            self._park(event, cleaned.rejection_code, cleaned.detail or "", stats)
            return

        outcome = self._loader.load(self._engine, [cleaned], run_id=event.run_id)
        stats.loaded += outcome.records_loaded

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
