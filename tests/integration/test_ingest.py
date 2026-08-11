"""The v0.1 ingestion run, end to end against real PostgreSQL.

Every stage is tested in isolation elsewhere. What this asserts is the seam:
that a run opens a record, resolves candidates, loads what it got, records why
each source was used, and closes the record on every path.
"""

from __future__ import annotations

import gzip
import json
from typing import Any

import pytest
from sqlalchemy import Connection, Engine, select

import pipeline.ingest as ingest_module
from pipeline.config import Settings
from pipeline.ingest import IngestReport, run_ingestion, run_resolution_to_sink
from pipeline.models.db import (
    books,
    ingestion_runs,
    rejected_records,
    resolution_attempts,
    source_runs,
)
from pipeline.models.domain import CandidateBook
from pipeline.models.events import EventType

pytestmark = pytest.mark.integration


@pytest.fixture
def offline(migrated_engine: Engine, tmp_path: Any) -> Settings:
    """Settings with every network source switched off by budget.

    The retained discovery payload costs nothing and needs no source, so the
    whole pipeline is exercisable without a single request.
    """
    return Settings(  # type: ignore[call-arg]
        database_url=str(migrated_engine.url).replace("***", "test"),
        openlibrary_contact_email="ci@example.com",
        discovery_manifest_path=tmp_path / "manifest.jsonl",
        openlibrary_max_fallback_queries_per_run=0,
        gutendex_max_last_resort_queries_per_run=0,
        googlebooks_enabled=False,
    )


def write_manifest(settings: Settings, count: int) -> None:
    with settings.discovery_manifest_path.open("w", encoding="utf-8") as out:
        for n in range(count):
            candidate = CandidateBook(
                candidate_key=f"/books/OL{n}M",
                title=f"Book {n}",
                isbns=[],
                discovery_payload={
                    "key": f"/books/OL{n}M",
                    "title": f"Book {n}",
                    "by_statement": f"by Author {n}",
                    "publish_date": "1998",
                },
            )
            out.write(json.dumps(candidate.model_dump(mode="json")) + "\n")


class TestReportStatus:
    def test_no_candidates_is_a_failure(self) -> None:
        # A run that found nothing to do did not succeed at doing it.
        assert IngestReport().status == "failed"

    def test_nothing_resolved_is_a_failure(self) -> None:
        assert IngestReport(candidates=10, unresolved=10).status == "failed"

    def test_everything_resolved_is_a_success(self) -> None:
        assert IngestReport(candidates=10, resolved=10).status == "success"

    def test_a_mixture_is_partial_success(self) -> None:
        # With a hierarchy of fallible sources this is the common case, and
        # rounding it either way throws away the number an operator wants.
        assert IngestReport(candidates=10, resolved=7, unresolved=3).status == ("partial_success")


class TestRun:
    def test_a_run_loads_the_catalogue(
        self, offline: Settings, engine: Engine, connection: Connection
    ) -> None:
        write_manifest(offline, 5)

        report = run_ingestion(offline, engine=engine)

        assert report.candidates == 5
        assert report.books_inserted == 5
        assert connection.execute(select(books.c.id)).scalars().all()

    def test_the_run_record_is_opened_and_closed(
        self, offline: Settings, engine: Engine, connection: Connection
    ) -> None:
        write_manifest(offline, 3)

        run_ingestion(offline, engine=engine)

        row = connection.execute(
            select(ingestion_runs.c.status, ingestion_runs.c.processing_ended_at)
        ).one()
        assert row.status == "success"
        assert row.processing_ended_at is not None

    def test_counts_land_on_the_run_record(
        self, offline: Settings, engine: Engine, connection: Connection
    ) -> None:
        write_manifest(offline, 4)

        run_ingestion(offline, engine=engine)

        row = connection.execute(select(ingestion_runs.c.records_loaded)).one()
        assert row.records_loaded == 4

    def test_every_source_attempt_is_recorded(
        self, offline: Settings, engine: Engine, connection: Connection
    ) -> None:
        write_manifest(offline, 2)

        run_ingestion(offline, engine=engine)

        assert connection.execute(select(resolution_attempts.c.source).distinct()).scalars().all()

    def test_a_disabled_goodreads_is_recorded_as_a_skipped_source(
        self, offline: Settings, engine: Engine, connection: Connection
    ) -> None:
        # A skipped source must be visible in the run record; inferring it from
        # a gap in the data is how a misconfiguration survives for weeks.
        write_manifest(offline, 1)

        run_ingestion(offline, engine=engine)

        row = connection.execute(
            select(source_runs.c.source, source_runs.c.status, source_runs.c.error)
        ).one()
        assert row.source == "goodreads"
        assert row.status == "skipped"
        assert "disabled" in row.error

    def test_rerunning_changes_nothing(
        self, offline: Settings, engine: Engine, connection: Connection
    ) -> None:
        write_manifest(offline, 5)
        run_ingestion(offline, engine=engine)
        before = connection.execute(select(books.c.updated_at)).scalars().all()
        connection.rollback()

        second = run_ingestion(offline, engine=engine)

        assert second.books_inserted == 0
        assert second.books_unchanged == 5
        assert connection.execute(select(books.c.updated_at)).scalars().all() == before

    def test_the_limit_is_honoured(self, offline: Settings, engine: Engine) -> None:
        write_manifest(offline, 10)

        report = run_ingestion(offline, limit=3, engine=engine)

        assert report.candidates == 3

    def test_a_missing_manifest_and_dump_fails_loudly(
        self, offline: Settings, engine: Engine
    ) -> None:
        # Silently discovering nothing would look like an empty catalogue.
        with pytest.raises(FileNotFoundError, match="nothing to discover"):
            run_ingestion(offline, engine=engine)

    def test_a_failed_run_still_closes_its_record(
        self, offline: Settings, engine: Engine, connection: Connection
    ) -> None:
        # A crashed run must be distinguishable from one that never started.
        with pytest.raises(FileNotFoundError):
            run_ingestion(offline, engine=engine)

        assert connection.execute(select(ingestion_runs.c.status)).scalar_one() == "failed"


class TestRemainingPaths:
    def test_the_dump_is_used_when_no_manifest_exists(
        self, offline: Settings, engine: Engine, connection: Connection, tmp_path: Any
    ) -> None:
        # The convenience path for a first run, before a manifest is built.
        dump = tmp_path / "dump.txt.gz"
        with gzip.open(dump, "wt", encoding="utf-8") as out:
            out.write(
                "\t".join(
                    [
                        "/type/edition",
                        "/books/OL1M",
                        "1",
                        "2026-01-01T00:00:00.000000",
                        json.dumps(
                            {
                                "key": "/books/OL1M",
                                "title": "From The Dump",
                                "isbn_13": ["9780441172719"],
                            }
                        ),
                    ]
                )
                + "\n"
            )
        direct = offline.model_copy(update={"openlibrary_dump_path": dump})

        report = run_ingestion(direct, engine=engine)

        assert report.books_inserted == 1
        assert connection.execute(select(books.c.title)).scalar_one() == "From The Dump"

    def test_an_unresolvable_candidate_is_counted_not_dropped(
        self, offline: Settings, engine: Engine
    ) -> None:
        # Every source is budgeted to zero and the payload is unusable, so the
        # candidate resolves to nothing — the run must still account for it.
        offline.discovery_manifest_path.write_text(
            json.dumps(
                CandidateBook(
                    candidate_key="/books/OL9M",
                    title="Unresolvable",
                    isbns=["9780441172719"],
                    discovery_payload={},
                ).model_dump(mode="json")
            )
            + "\n"
        )

        report = run_ingestion(offline, engine=engine)

        assert report.candidates == 1
        assert report.unresolved == 1
        assert report.status == "failed"

    def test_a_payload_that_fails_validation_is_recorded_as_a_rejection(
        self, offline: Settings, engine: Engine, connection: Connection
    ) -> None:
        offline.discovery_manifest_path.write_text(
            json.dumps(
                CandidateBook(
                    candidate_key="/books/OL8M",
                    title="Bad Payload",
                    isbns=["9780441172719"],
                    discovery_payload={"no_key_here": True},
                ).model_dump(mode="json")
            )
            + "\n"
        )

        run_ingestion(offline, engine=engine)

        assert connection.execute(select(rejected_records.c.stage)).scalar_one() == ("extract")


class TestResolutionToSink:
    """Phase 2's extract stage.

    The same discovery and resolution, but observations go onto a topic instead
    of into the catalogue. Attempts and rejections still belong here — a
    consumer has no way to reconstruct why a source was skipped.
    """

    def _sink(self) -> Any:
        class Recording:
            def __init__(self) -> None:
                self.emitted: list[Any] = []
                self.flushes = 0

            def emit(self, records: Any) -> None:
                self.emitted.extend(records)

            def flush(self) -> None:
                self.flushes += 1

        return Recording()

    def test_observations_are_produced_not_loaded(
        self, offline: Settings, engine: Engine, connection: Connection
    ) -> None:
        write_manifest(offline, 4)
        sink = self._sink()

        report = run_resolution_to_sink(offline, sink, engine=engine)

        assert report.observations == 4
        assert len(sink.emitted) == 4
        # The catalogue is the consumers' job now.
        assert connection.execute(select(books.c.id)).scalars().all() == []

    def test_the_events_are_raw_and_carry_the_run(self, offline: Settings, engine: Engine) -> None:
        write_manifest(offline, 1)
        sink = self._sink()

        report = run_resolution_to_sink(offline, sink, engine=engine)

        assert sink.emitted[0].event_type is EventType.BOOK_RAW
        assert sink.emitted[0].run_id == report.run_id

    def test_each_batch_is_flushed(self, offline: Settings, engine: Engine) -> None:
        # A crash then costs one batch of re-resolved candidates rather than
        # the whole run's external calls.
        write_manifest(offline, 3)
        sink = self._sink()

        run_resolution_to_sink(offline, sink, engine=engine)

        assert sink.flushes >= 1

    def test_attempts_are_still_recorded(
        self, offline: Settings, engine: Engine, connection: Connection
    ) -> None:
        write_manifest(offline, 2)

        run_resolution_to_sink(offline, self._sink(), engine=engine)

        assert connection.execute(select(resolution_attempts.c.source)).scalars().all()

    def test_the_run_is_opened_and_its_id_returned(
        self, offline: Settings, engine: Engine, connection: Connection
    ) -> None:
        # The barrier needs it to know which run to close.
        write_manifest(offline, 1)

        report = run_resolution_to_sink(offline, self._sink(), engine=engine)

        assert report.run_id is not None
        assert (
            connection.execute(
                select(ingestion_runs.c.id).where(ingestion_runs.c.id == report.run_id)
            ).scalar_one()
            == report.run_id
        )

    def test_a_failure_closes_the_run(
        self, offline: Settings, engine: Engine, connection: Connection
    ) -> None:
        # A crashed run must be distinguishable from one that never started.
        with pytest.raises(FileNotFoundError):
            run_resolution_to_sink(offline, self._sink(), engine=engine)

        assert connection.execute(select(ingestion_runs.c.status)).scalar_one() == "failed"

    def test_the_limit_is_honoured(self, offline: Settings, engine: Engine) -> None:
        write_manifest(offline, 8)

        report = run_resolution_to_sink(offline, self._sink(), limit=3, engine=engine)

        assert report.candidates == 3


class TestIdentityConflictHandling:
    def test_two_different_books_behind_one_candidate_are_loaded_apart(
        self, offline: Settings, engine: Engine
    ) -> None:
        """Observations that disagree on ISBN are not the same book.

        Forcing them onto one identity would fuse a pair nothing could separate
        again, so the run loads them separately and says so in the log rather
        than guessing which ISBN is right.
        """
        # Two clean records for one candidate, carrying different ISBNs.
        original = ingest_module.unify_identity

        def conflicting(candidates: Any) -> Any:
            if len(candidates) > 1:
                msg = "observations disagree on ISBN"
                raise ValueError(msg)
            return original(candidates)

        ingest_module.unify_identity = conflicting  # type: ignore[assignment]
        try:
            write_manifest(offline, 1)
            report = run_ingestion(offline, engine=engine)
        finally:
            ingest_module.unify_identity = original  # type: ignore[assignment]

        assert report.candidates == 1


class TestUnresolvedCandidates:
    def test_a_candidate_resolving_to_nothing_is_counted_by_the_produce_path(
        self, offline: Settings, engine: Engine
    ) -> None:
        # Every source budgeted to zero and no usable discovery payload.
        offline.discovery_manifest_path.write_text(
            json.dumps(
                CandidateBook(
                    candidate_key="/books/OL7M",
                    title="Nothing Resolves This",
                    isbns=["9780441172719"],
                    discovery_payload={},
                ).model_dump(mode="json")
            )
            + "\n"
        )

        sink_records: list[Any] = []

        class Recording:
            def emit(self, records: Any) -> None:
                sink_records.extend(records)

            def flush(self) -> None:
                return None

        report = run_resolution_to_sink(offline, Recording(), engine=engine)

        assert report.unresolved == 1
        assert report.observations == 0
        assert sink_records == []
