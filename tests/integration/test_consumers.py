"""The consumer services against real PostgreSQL.

The claims under test are the ones the phase-2 design rests on: redelivery
changes nothing, a poison record is parked rather than blocking a partition,
and a consumer that restarts mid-run finishes the run instead of stalling.

The file adapters stand in for Kafka here. That is the whole point of the
Source/Sink contracts — these services cannot tell the difference, so their
behaviour can be pinned without a broker. Kafka's own guarantees are exercised
separately.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Connection, Engine, insert, select

from pipeline.messaging import FileSink, FileSource
from pipeline.models.db import books, ingestion_runs
from pipeline.models.domain import SourceName
from pipeline.models.events import BookEvent, EventType, PartitionMarker
from pipeline.observability.markers import freeze_topology, mark_processing
from pipeline.services import LoadConsumer, TransformConsumer

pytestmark = pytest.mark.integration

PARTITIONS = 3


@pytest.fixture
def run_id(engine: Engine) -> uuid.UUID:
    """A run already handed over to the consumers."""
    identifier = uuid.uuid4()
    with engine.begin() as connection:
        connection.execute(
            insert(ingestion_runs).values(id=identifier, dag_run_id=f"cli:{identifier}")
        )
        freeze_topology(connection, identifier, "books.raw", PARTITIONS)
        freeze_topology(connection, identifier, "books.clean", PARTITIONS)
        mark_processing(connection, identifier)
    return identifier


def raw_event(run_id: uuid.UUID, source_id: str = "1", **payload: Any) -> BookEvent:
    return BookEvent(
        run_id=run_id,
        source=SourceName.GUTENDEX,
        source_id=source_id,
        payload={"id": int(source_id), "title": f"Book {source_id}", **payload},
    )


def clean_event(run_id: uuid.UUID, source_id: str = "1") -> BookEvent:
    return BookEvent(
        run_id=run_id,
        source=SourceName.GUTENDEX,
        source_id=source_id,
        event_type=EventType.BOOK_CLEAN,
        identity_key="fallback:" + "a" * 64,
        payload={"id": int(source_id), "title": f"Book {source_id}"},
    )


def markers(run_id: uuid.UUID, topic: str) -> list[PartitionMarker]:
    return [PartitionMarker(run_id=run_id, topic=topic, partition=n) for n in range(PARTITIONS)]


def run_transform(
    engine: Engine, tmp_path: Path, events: list[Any], **kwargs: Any
) -> tuple[Any, Path, Path]:
    source_path, clean_path, dlq_path = (
        tmp_path / "raw.jsonl",
        tmp_path / "clean.jsonl",
        tmp_path / "dlq.jsonl",
    )
    FileSink(source_path).emit(events)
    consumer = TransformConsumer(
        engine,
        FileSource(source_path),
        FileSink(clean_path),
        FileSink(dlq_path),
        clean_partitions=PARTITIONS,
        **kwargs,
    )
    return consumer.run(), clean_path, dlq_path


class TestTransformConsumer:
    def test_a_raw_event_becomes_a_clean_one(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        stats, clean_path, _ = run_transform(engine, tmp_path, [raw_event(run_id)])

        assert stats.transformed == 1
        assert len(clean_path.read_text().splitlines()) == 1

    def test_the_clean_event_carries_canonical_identity(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        _, clean_path, _ = run_transform(engine, tmp_path, [raw_event(run_id)])

        emitted = list(FileSource(clean_path).consume())

        assert isinstance(emitted[0], BookEvent)
        assert emitted[0].event_type is EventType.BOOK_CLEAN
        assert emitted[0].identity_key is not None

    def test_an_unusable_record_is_parked_not_dropped(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        # A record that fails validation cannot be retried into correctness.
        stats, _clean, dlq_path = run_transform(engine, tmp_path, [raw_event(run_id, title="")])

        assert stats.rejected == 1
        assert stats.transformed == 0
        assert len(dlq_path.read_text().splitlines()) == 1

    def test_a_parked_record_says_why(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        # A DLQ entry that does not say what went wrong is a queue nobody drains.
        _, _, dlq_path = run_transform(engine, tmp_path, [raw_event(run_id, title="")])

        parked = list(FileSource(dlq_path).consume())

        assert isinstance(parked[0], BookEvent)
        assert parked[0].payload["_failure"]["code"]
        assert parked[0].payload["_failure"]["attempts"] >= 1

    def test_one_poison_record_does_not_block_the_rest(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        stats, _, _ = run_transform(
            engine,
            tmp_path,
            [raw_event(run_id, "1"), raw_event(run_id, "2", title=""), raw_event(run_id, "3")],
        )

        assert stats.transformed == 2
        assert stats.rejected == 1

    def test_all_raw_markers_forward_the_boundary(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        stats, clean_path, _ = run_transform(
            engine, tmp_path, [raw_event(run_id), *markers(run_id, "books.raw")]
        )

        forwarded = [e for e in FileSource(clean_path).consume() if isinstance(e, PartitionMarker)]
        assert stats.runs_completed == 1
        assert len(forwarded) == PARTITIONS

    def test_a_partial_set_of_markers_forwards_nothing(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        # The boundary is all partitions or none of them.
        stats, clean_path, _ = run_transform(engine, tmp_path, markers(run_id, "books.raw")[:2])

        assert stats.runs_completed == 0
        assert not [e for e in FileSource(clean_path).consume() if isinstance(e, PartitionMarker)]


class TestLoadConsumer:
    def _load(self, engine: Engine, tmp_path: Path, events: list[Any]) -> tuple[Any, Path]:
        clean_path, dlq_path = tmp_path / "clean.jsonl", tmp_path / "dlq.jsonl"
        FileSink(clean_path).emit(events)
        consumer = LoadConsumer(engine, FileSource(clean_path), FileSink(dlq_path))
        return consumer.run(), dlq_path

    def test_a_clean_event_reaches_the_catalogue(
        self, engine: Engine, connection: Connection, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        self._load(engine, tmp_path, [clean_event(run_id)])

        assert connection.execute(select(books.c.title)).scalar_one() == "Book 1"

    def test_redelivery_changes_nothing(
        self, engine: Engine, connection: Connection, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        # The claim the whole at-least-once design rests on: a crash between
        # the database commit and the offset commit replays the record, and
        # replaying must be free.
        self._load(engine, tmp_path, [clean_event(run_id)] * 3)

        rows = connection.execute(select(books.c.id, books.c.updated_at)).all()
        assert len(rows) == 1

    def test_all_clean_markers_finalise_the_run(
        self, engine: Engine, connection: Connection, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        stats, _ = self._load(
            engine, tmp_path, [clean_event(run_id), *markers(run_id, "books.clean")]
        )

        assert stats.runs_finalised == 1
        assert (
            connection.execute(
                select(ingestion_runs.c.status).where(ingestion_runs.c.id == run_id)
            ).scalar_one()
            == "success"
        )

    def test_a_redelivered_marker_does_not_refinalise(
        self, engine: Engine, connection: Connection, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        # Guarded on status = 'processing', so a replay cannot re-time a run.
        self._load(engine, tmp_path, markers(run_id, "books.clean"))
        first = connection.execute(
            select(ingestion_runs.c.processing_ended_at).where(ingestion_runs.c.id == run_id)
        ).scalar_one()
        connection.rollback()

        stats, _ = self._load(engine, tmp_path, markers(run_id, "books.clean"))

        assert stats.runs_finalised == 0
        assert (
            connection.execute(
                select(ingestion_runs.c.processing_ended_at).where(ingestion_runs.c.id == run_id)
            ).scalar_one()
            == first
        )


class TestRestartRecovery:
    """The failure that is invisible without durable markers.

    A consumer that tracked markers in memory would commit offsets past them,
    resume after them on restart, and leave the run in `processing` forever —
    with every book correctly loaded, which is exactly what makes it so easy to
    miss.
    """

    def test_a_load_consumer_restarting_mid_run_still_finalises(
        self, engine: Engine, connection: Connection, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        first_batch = tmp_path / "clean-1.jsonl"
        FileSink(first_batch).emit(markers(run_id, "books.clean")[:2])
        LoadConsumer(engine, FileSource(first_batch), FileSink(tmp_path / "d1.jsonl")).run()

        assert (
            connection.execute(
                select(ingestion_runs.c.status).where(ingestion_runs.c.id == run_id)
            ).scalar_one()
            == "processing"
        )
        connection.rollback()

        # A brand new consumer: nothing carried over but the marker rows.
        last_batch = tmp_path / "clean-2.jsonl"
        FileSink(last_batch).emit(markers(run_id, "books.clean")[2:])
        stats = LoadConsumer(engine, FileSource(last_batch), FileSink(tmp_path / "d2.jsonl")).run()

        assert stats.runs_finalised == 1
        assert (
            connection.execute(
                select(ingestion_runs.c.status).where(ingestion_runs.c.id == run_id)
            ).scalar_one()
            == "success"
        )

    def test_the_startup_sweep_finishes_a_run_whose_markers_all_arrived(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        # Every marker seen, then the process died before finalising. A fresh
        # consumer with an empty source must still close the run.
        seen = tmp_path / "clean.jsonl"
        FileSink(seen).emit(markers(run_id, "books.clean"))
        LoadConsumer(engine, FileSource(seen), FileSink(tmp_path / "d.jsonl")).run()
        with engine.begin() as conn:
            conn.execute(
                ingestion_runs.update()
                .where(ingestion_runs.c.id == run_id)
                .values(status="processing")
            )

        stats = LoadConsumer(
            engine, FileSource(tmp_path / "empty.jsonl"), FileSink(tmp_path / "d2.jsonl")
        ).run()

        assert stats.runs_finalised == 1

    def test_a_transform_consumer_restarting_mid_run_still_forwards(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        first = tmp_path / "raw-1.jsonl"
        FileSink(first).emit(markers(run_id, "books.raw")[:2])
        TransformConsumer(
            engine,
            FileSource(first),
            FileSink(tmp_path / "c1.jsonl"),
            FileSink(tmp_path / "d1.jsonl"),
            clean_partitions=PARTITIONS,
        ).run()

        last = tmp_path / "raw-2.jsonl"
        clean_path = tmp_path / "c2.jsonl"
        FileSink(last).emit(markers(run_id, "books.raw")[2:])
        stats = TransformConsumer(
            engine,
            FileSource(last),
            FileSink(clean_path),
            FileSink(tmp_path / "d2.jsonl"),
            clean_partitions=PARTITIONS,
        ).run()

        assert stats.runs_completed == 1
        assert (
            len([e for e in FileSource(clean_path).consume() if isinstance(e, PartitionMarker)])
            == PARTITIONS
        )


class TestDeadLetterPaths:
    """Records neither consumer can use.

    Both park rather than drop, and both flush before returning: if the DLQ
    write fails the exception propagates and the offset is never committed, so
    the record is redelivered instead of lost.
    """

    def test_the_load_consumer_parks_an_unmappable_payload(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        clean_path, dlq_path = tmp_path / "clean.jsonl", tmp_path / "dlq.jsonl"
        FileSink(clean_path).emit(
            [
                BookEvent(
                    run_id=run_id,
                    source=SourceName.GUTENDEX,
                    source_id="9",
                    event_type=EventType.BOOK_CLEAN,
                    identity_key="fallback:" + "b" * 64,
                    payload={"nothing": "usable"},
                )
            ]
        )

        stats = LoadConsumer(engine, FileSource(clean_path), FileSink(dlq_path)).run()

        assert stats.rejected == 1
        assert stats.loaded == 0
        assert len(dlq_path.read_text().splitlines()) == 1

    def test_the_load_consumer_parks_a_record_that_cannot_be_canonicalised(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        clean_path, dlq_path = tmp_path / "clean.jsonl", tmp_path / "dlq.jsonl"
        FileSink(clean_path).emit(
            [
                BookEvent(
                    run_id=run_id,
                    source=SourceName.GUTENDEX,
                    source_id="9",
                    event_type=EventType.BOOK_CLEAN,
                    identity_key="fallback:" + "c" * 64,
                    payload={"id": 9, "title": "   "},
                )
            ]
        )

        stats = LoadConsumer(engine, FileSource(clean_path), FileSink(dlq_path)).run()

        assert stats.rejected == 1

    def test_a_marker_for_a_run_with_no_frozen_topology_completes_nothing(
        self, engine: Engine, tmp_path: Path
    ) -> None:
        # The barrier has not run for this topic, so there is no expectation to
        # compare against and no boundary to declare.
        orphan = uuid.uuid4()
        with engine.begin() as connection:
            connection.execute(insert(ingestion_runs).values(id=orphan, dag_run_id=f"cli:{orphan}"))

        clean_path = tmp_path / "clean.jsonl"
        FileSink(clean_path).emit(
            [PartitionMarker(run_id=orphan, topic="books.clean", partition=0)]
        )
        stats = LoadConsumer(engine, FileSource(clean_path), FileSink(tmp_path / "d.jsonl")).run()

        assert stats.runs_finalised == 0


class TestOffsetCommits:
    """The commit the whole at-least-once story depends on.

    A consumer that processes without committing loses nothing, so every test
    of correctness still passes — and then replays the entire topic on every
    restart. A live run found exactly that: both groups had processed
    everything and committed nothing.
    """

    class CountingSource:
        """A finite source that records when the stage committed."""

        def __init__(self, events: list[Any]) -> None:
            self._events = events
            self.commits = 0

        def consume(self) -> Any:
            yield from self._events

        def commit(self) -> None:
            self.commits += 1

    def test_the_load_consumer_commits_after_each_event(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        source = self.CountingSource([clean_event(run_id, "1"), clean_event(run_id, "2")])

        LoadConsumer(engine, source, FileSink(tmp_path / "d.jsonl")).run()

        assert source.commits == 2

    def test_the_transform_consumer_commits_after_each_event(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        source = self.CountingSource([raw_event(run_id, "1"), raw_event(run_id, "2")])

        TransformConsumer(
            engine,
            source,
            FileSink(tmp_path / "c.jsonl"),
            FileSink(tmp_path / "d.jsonl"),
            clean_partitions=PARTITIONS,
        ).run()

        assert source.commits == 2

    def test_markers_are_committed_too(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        # An uncommitted marker would be re-observed on every restart, which
        # the primary key absorbs — but the topic would never drain.
        source = self.CountingSource(markers(run_id, "books.clean"))

        LoadConsumer(engine, source, FileSink(tmp_path / "d.jsonl")).run()

        assert source.commits == PARTITIONS

    def test_a_parked_record_is_still_committed(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        # It reached the DLQ, so it has been dealt with. Not committing would
        # park it again on every restart, forever.
        source = self.CountingSource([raw_event(run_id, "1", title="")])

        TransformConsumer(
            engine,
            source,
            FileSink(tmp_path / "c.jsonl"),
            FileSink(tmp_path / "d.jsonl"),
            clean_partitions=PARTITIONS,
        ).run()

        assert source.commits == 1
