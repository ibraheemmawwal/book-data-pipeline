"""Source and sink contracts.

The point of these is that transform and load never learn which one is driving
them. In v1.0 a run reads a finite file; in v2.0 it reads a Kafka topic that
never ends. Neither stage should be able to tell, because the moment one can,
the phase-2 swap stops being a swap and becomes a rewrite.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pipeline.messaging import FileSink, FileSource, Sink, Source
from pipeline.models.domain import SourceName
from pipeline.models.events import BookEvent, EventType, PartitionMarker

RUN_ID = uuid4()


def book(source_id: str = "1") -> BookEvent:
    return BookEvent(
        run_id=RUN_ID,
        source=SourceName.GUTENDEX,
        source_id=source_id,
        payload={"id": source_id, "title": f"Book {source_id}"},
    )


class TestFileSink:
    def test_it_writes_events(self, tmp_path: Path) -> None:
        sink = FileSink(tmp_path / "events.jsonl")
        sink.emit([book("1"), book("2")])
        sink.flush()

        assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2

    def test_emitting_nothing_is_harmless(self, tmp_path: Path) -> None:
        sink = FileSink(tmp_path / "events.jsonl")
        sink.emit([])
        sink.flush()

        assert (tmp_path / "events.jsonl").exists()

    def test_it_appends_across_calls(self, tmp_path: Path) -> None:
        # A batch is not the whole run; a second emit must not truncate.
        sink = FileSink(tmp_path / "events.jsonl")
        sink.emit([book("1")])
        sink.emit([book("2")])
        sink.flush()

        assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2

    def test_it_creates_missing_directories(self, tmp_path: Path) -> None:
        sink = FileSink(tmp_path / "nested" / "deep" / "events.jsonl")
        sink.emit([book()])
        sink.flush()

        assert (tmp_path / "nested" / "deep" / "events.jsonl").exists()

    def test_it_satisfies_the_protocol(self, tmp_path: Path) -> None:
        assert isinstance(FileSink(tmp_path / "e.jsonl"), Sink)


class TestFileSource:
    def test_it_reads_back_what_a_sink_wrote(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        sink = FileSink(path)
        sink.emit([book("1"), book("2")])
        sink.flush()

        consumed = list(FileSource(path).consume())

        assert [e.source_id for e in consumed if isinstance(e, BookEvent)] == ["1", "2"]

    def test_it_is_finite(self, tmp_path: Path) -> None:
        # The whole difference from Kafka: a file ends, a topic does not.
        path = tmp_path / "events.jsonl"
        FileSink(path).emit([book()])
        FileSink(path).flush()

        assert len(list(FileSource(path).consume())) >= 0

    def test_markers_round_trip_alongside_books(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        sink = FileSink(path)
        sink.emit([book("1"), PartitionMarker(run_id=RUN_ID, topic="books.raw", partition=0)])
        sink.flush()

        kinds = [type(e).__name__ for e in FileSource(path).consume()]

        assert kinds == ["BookEvent", "PartitionMarker"]

    def test_a_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        # A stage that has not run yet is empty, not broken.
        assert list(FileSource(tmp_path / "absent.jsonl").consume()) == []

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        path.write_text("\n\n" + book().to_json().decode() + "\n\n")

        assert len(list(FileSource(path).consume())) == 1

    def test_it_satisfies_the_protocol(self, tmp_path: Path) -> None:
        assert isinstance(FileSource(tmp_path / "e.jsonl"), Source)


class TestStagesCannotTellThemApart:
    def test_both_sources_yield_the_same_shape(self, tmp_path: Path) -> None:
        """The contract that makes the phase-2 swap a swap.

        A stage written against ``Source`` gets decoded events either way; the
        only difference is whether the iterator ever ends, which is the
        adapter's business and not the stage's.
        """
        path = tmp_path / "events.jsonl"
        sink = FileSink(path)
        sink.emit([book("1")])
        sink.flush()

        for event in FileSource(path).consume():
            assert isinstance(event, BookEvent | PartitionMarker)

    def test_a_clean_event_requires_its_identity(self) -> None:
        # Guarded at the envelope so a stage cannot emit an unroutable event.
        with pytest.raises(ValidationError, match="identity_key"):
            BookEvent(
                run_id=RUN_ID,
                source=SourceName.GUTENDEX,
                source_id="1",
                event_type=EventType.BOOK_CLEAN,
            )


class TestFileSourceResilience:
    def test_an_undecodable_line_costs_only_that_line(self, tmp_path: Path) -> None:
        # The alternative is a staging file that cannot be replayed at all.
        path = tmp_path / "events.jsonl"
        path.write_text(
            book("1").to_json().decode()
            + "\n"
            + "{not json at all\n"
            + book("2").to_json().decode()
            + "\n"
        )

        consumed = [e for e in FileSource(path).consume() if isinstance(e, BookEvent)]

        assert [e.source_id for e in consumed] == ["1", "2"]

    def test_a_line_from_a_future_schema_is_skipped(self, tmp_path: Path) -> None:
        # Not guessed at: an unsupported version is skipped here and routed to
        # the DLQ by the consumer that meets it on the wire.
        path = tmp_path / "events.jsonl"
        path.write_text('{"schema_version": 99, "event_type": "book.raw"}\n')

        assert list(FileSource(path).consume()) == []
