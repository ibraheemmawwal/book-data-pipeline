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
from sqlalchemy import Connection, Engine, func, insert, select

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


def count(engine: Engine, table: Any) -> int:
    """Rows in a table, through a connection of its own.

    Its own connection on purpose: it is called from inside a commit hook, so
    it must not depend on a transaction the consumer is in the middle of.
    """
    with engine.connect() as connection:
        return int(connection.execute(select(func.count()).select_from(table)).scalar_one())


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

    def test_the_load_consumer_commits_what_it_wrote(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        # Not "once per event" — the consumer writes in batches, so the
        # guarantee is that it commits at all, and only after writing.
        source = self.CountingSource([clean_event(run_id, "1"), clean_event(run_id, "2")])

        LoadConsumer(engine, source, FileSink(tmp_path / "d.jsonl")).run()

        assert source.commits >= 1
        assert count(engine, books) == 2

    def test_batching_commits_once_for_the_whole_batch(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        """The point of batching, stated as a number.

        One transaction per book spends two of its fourteen round trips on
        BEGIN and COMMIT, which is invisible locally and 15% of the bill
        against a database an ocean away.
        """
        events = [clean_event(run_id, str(n)) for n in range(20)]
        source = self.CountingSource(events)

        LoadConsumer(engine, source, FileSink(tmp_path / "d.jsonl"), batch_size=20).run()

        assert source.commits == 1, f"{source.commits} commits for one batch of 20"
        assert count(engine, books) == 20

    def test_nothing_is_committed_before_it_is_written(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        """The ordering the whole at-least-once story rests on.

        Committing first turns a crash into silent data loss; committing after
        turns it into a replay, which the idempotent load makes harmless.
        """
        written_at_commit: list[int] = []

        class Watching(self.CountingSource):  # type: ignore[name-defined,misc]
            def commit(inner) -> None:  # noqa: N805
                written_at_commit.append(count(engine, books))
                super().commit()

        source = Watching([clean_event(run_id, str(n)) for n in range(6)])
        LoadConsumer(engine, source, FileSink(tmp_path / "d.jsonl"), batch_size=3).run()

        assert written_at_commit, "never committed"
        # Every commit happened with all preceding records already in the
        # database; a commit observing fewer rows would be a commit ahead of
        # its own write.
        assert written_at_commit == sorted(written_at_commit)
        assert written_at_commit[-1] == 6

    def test_a_batch_is_written_before_the_marker_that_follows_it(
        self, engine: Engine, tmp_path: Path, run_id: uuid.UUID
    ) -> None:
        """A run must not be declared complete while its books are pending.

        The marker is what finalises the run. With batching, records can still
        be sitting unwritten when it arrives, so the flush has to happen first.

        Asserting the final row count cannot see this: the run ends with a
        flush either way, and by the time the test looks, the books are in. The
        only discriminating question is what was already written *at the moment
        the marker was handled*.
        """
        seen: list[int] = []
        events: list[Any] = [clean_event(run_id, str(n)) for n in range(5)]
        events += markers(run_id, "books.clean")

        # Larger than the run, so nothing flushes on size alone.
        consumer = LoadConsumer(
            engine, self.CountingSource(events), FileSink(tmp_path / "d.jsonl"), batch_size=1000
        )
        original = consumer._handle_marker

        def watching(event: Any, stats: Any) -> None:
            seen.append(count(engine, books))
            original(event, stats)

        consumer._handle_marker = watching  # type: ignore[method-assign]
        stats = consumer.run()

        assert seen, "no marker was handled"
        assert seen[0] == 5, (
            f"the first marker was handled with only {seen[0]} of 5 books written; "
            "the run can be finalised before its records exist"
        )
        assert stats.runs_finalised == 1
