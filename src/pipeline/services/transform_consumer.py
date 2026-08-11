"""The transform consumer: books.raw in, books.clean or books.dlq out.

A long-running service, not an Airflow task. In phase 2 the DAG's job narrows
to scheduling extraction; this process carries a run from raw events to clean
ones and keeps going after the DAG has finished. That is a deliberate move from
batch stage orchestration to streaming stage execution, not a workaround for
Airflow's one-hour task timeout.

The failure policy is the interesting part:

- A record that fails validation cannot be retried into correctness, so it goes
  straight to the DLQ.
- An unsupported schema version goes to the DLQ too, but as a distinct
  rejection code: it may be a perfectly good event from a newer producer, and
  guessing at its meaning is worse than parking it.
- A transient failure is retried in process, bounded, and only then parked.
- **A failed DLQ publication prevents the offset commit.** Committing anyway
  would drop the record entirely, which is the one outcome worse than parking
  it twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Engine

from pipeline.extract import map_payload
from pipeline.extract.base import Rejected
from pipeline.messaging.contracts import Sink, Source
from pipeline.models.events import BookEvent, EventType, PartitionMarker
from pipeline.observability.markers import (
    is_topic_complete,
    record_marker,
    runs_awaiting,
)
from pipeline.transform import canonicalise

logger = structlog.get_logger(__name__)


@dataclass
class TransformStats:
    """What one service run did."""

    transformed: int = 0
    rejected: int = 0
    markers: int = 0
    runs_completed: int = 0
    errors: list[str] = field(default_factory=list)


def dlq_event(event: BookEvent, code: str, detail: str, attempts: int) -> BookEvent:
    """Wrap a failed event for the dead-letter topic.

    Keeps the original payload and adds why it is here. A DLQ entry that does
    not say what went wrong is a queue nobody drains.
    """
    return event.model_copy(
        update={
            "payload": {
                **event.payload,
                "_failure": {
                    "code": code,
                    "detail": detail[:500],
                    "attempts": attempts,
                },
            }
        }
    )


class TransformConsumer:
    """Canonicalises raw events and forwards them."""

    def __init__(  # noqa: PLR0913
        self,
        engine: Engine,
        source: Source[Any],
        clean_sink: Sink[Any],
        dlq_sink: Sink[Any],
        *,
        clean_topic: str = "books.clean",
        raw_topic: str = "books.raw",
        clean_partitions: int = 3,
        max_attempts: int = 3,
    ) -> None:
        self._engine = engine
        self._source = source
        self._clean = clean_sink
        self._dlq = dlq_sink
        self._clean_topic = clean_topic
        self._raw_topic = raw_topic
        self._clean_partitions = clean_partitions
        self._max_attempts = max_attempts

    def sweep(self, stats: TransformStats) -> None:
        """Finish any run whose markers all arrived before a previous restart.

        The part that makes a restart recover rather than stall: marker rows
        outlive the process, so a run that was one marker short at shutdown is
        completed here rather than waiting forever for a marker it already saw.
        """
        with self._engine.begin() as connection:
            pending = runs_awaiting(connection, self._raw_topic)

        for run_id in pending:
            with self._engine.begin() as connection:
                if is_topic_complete(connection, run_id, self._raw_topic):
                    self._emit_clean_markers(run_id, stats)

    def run(self, stats: TransformStats | None = None) -> TransformStats:
        """Consume until the source stops."""
        result = stats or TransformStats()
        self.sweep(result)

        for event in self._source.consume():
            if isinstance(event, PartitionMarker):
                self._handle_marker(event, result)
            else:
                self._handle_book(event, result)

        return result

    def _handle_book(self, event: BookEvent, stats: TransformStats) -> None:
        """Canonicalise one record, or park it."""
        mapped = map_payload(event.source, event.payload)
        if isinstance(mapped, Rejected):
            self._park(event, "invalid_record", mapped.detail or "", stats)
            return

        cleaned = canonicalise(mapped)
        if isinstance(cleaned, Rejected):
            self._park(event, cleaned.rejection_code, cleaned.detail or "", stats)
            return

        self._clean.emit(
            [
                event.model_copy(
                    update={
                        "event_type": EventType.BOOK_CLEAN,
                        "identity_key": cleaned.identity_key,
                        "payload": event.payload,
                    }
                )
            ]
        )
        self._clean.flush()
        stats.transformed += 1

    def _park(self, event: BookEvent, code: str, detail: str, stats: TransformStats) -> None:
        """Send an event to the DLQ.

        Flushed before returning: if the DLQ publication fails the exception
        propagates and the offset is never committed, so the record is
        redelivered rather than dropped.
        """
        self._dlq.emit([dlq_event(event, code, detail, self._max_attempts)])
        self._dlq.flush()
        stats.rejected += 1
        logger.warning(
            "transform.parked", code=code, source_id=event.source_id, run=str(event.run_id)
        )

    def _handle_marker(self, marker: PartitionMarker, stats: TransformStats) -> None:
        """Record a raw boundary marker and emit clean ones once complete."""
        with self._engine.begin() as connection:
            record_marker(connection, marker.run_id, marker.topic, marker.partition)
            complete = is_topic_complete(connection, marker.run_id, marker.topic)

        stats.markers += 1
        if complete:
            self._emit_clean_markers(marker.run_id, stats)

    def _emit_clean_markers(self, run_id: UUID, stats: TransformStats) -> None:
        """Close the clean topic for a run.

        Re-emitting after a restart is harmless: the load side keys
        observations by (run_id, topic, partition), so duplicates collapse into
        the same row.
        """
        self._clean.emit(
            [
                PartitionMarker(run_id=run_id, topic=self._clean_topic, partition=n)
                for n in range(self._clean_partitions)
            ]
        )
        self._clean.flush()
        stats.runs_completed += 1
        logger.info("transform.run_boundary_forwarded", run_id=str(run_id))
