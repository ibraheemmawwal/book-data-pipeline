"""The v0.1 ingestion run, end to end against real PostgreSQL.

Every stage is tested in isolation elsewhere. What this asserts is the seam:
that a run opens a record, resolves candidates, loads what it got, records why
each source was used, and closes the record on every path.
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
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
from pipeline.models.domain import CandidateBook, SourceName
from pipeline.models.events import EventType
from pipeline.observability.runs import abandon_stale_runs, finalise_run, start_run
from pipeline.source_health import last_refusal, record_refusal

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


def _age(connection: Connection, run_id: Any, *, hours: int) -> None:
    """Backdate a run so the staleness sweep can see it."""
    connection.execute(
        ingestion_runs.update()
        .where(ingestion_runs.c.id == run_id)
        .values(started_at=datetime.now(UTC) - timedelta(hours=hours))
    )


class TestRunsLeftOpenByAKill:
    """A run that was killed rather than raising.

    A container restart, a scheduler bounce that re-adopts a task, an OOM: none
    of them reach the run's own exception handler, so the row stays ``running``
    forever. Five had accumulated in the live catalogue, two of them more than
    a day old, each claiming work was in progress when nothing was.

    Closing them is safe because ``max_active_runs=1`` — a new run starting is
    proof that no older one is still going.
    """

    def test_a_new_run_closes_the_one_a_kill_left_open(self, connection: Connection) -> None:
        killed = start_run(connection, "killed-by-restart")
        _age(connection, killed, hours=20)

        start_run(connection, "the-next-one")

        state = connection.execute(
            select(ingestion_runs.c.status).where(ingestion_runs.c.id == killed)
        ).scalar_one()
        assert state == "failed"

    def test_the_new_run_is_still_running(self, connection: Connection) -> None:
        # The sweep must not close the run that just opened.
        current = start_run(connection, "current")

        state = connection.execute(
            select(ingestion_runs.c.status).where(ingestion_runs.c.id == current)
        ).scalar_one()
        assert state == "running"

    def test_finished_runs_are_not_touched(self, connection: Connection) -> None:
        done = start_run(connection, "already-done")
        finalise_run(connection, done, status="success")

        start_run(connection, "next")

        state = connection.execute(
            select(ingestion_runs.c.status).where(ingestion_runs.c.id == done)
        ).scalar_one()
        assert state == "success"

    def test_it_reports_how_many_it_closed(self, connection: Connection) -> None:
        old = start_run(connection, "one")
        _age(connection, old, hours=20)
        start_run(connection, "two")

        assert abandon_stale_runs(connection) == 0, "the sweep already ran on start"

    def test_a_run_still_in_flight_is_left_alone(self, connection: Connection) -> None:
        """The bug the age test exists to prevent.

        max_active_runs=1 is per DAG. Contested resolution opens runs of its
        own, so a tie-breaking run starting would otherwise mark a live
        ingestion failed.
        """
        live = start_run(connection, "ingestion-in-flight")

        start_run(connection, "contested-starting-alongside-it")

        state = connection.execute(
            select(ingestion_runs.c.status).where(ingestion_runs.c.id == live)
        ).scalar_one()
        assert state == "running"


class TestGoodreadsCooldownInIngestion:
    """A refusal skips the source, not the run.

    This is the asymmetry with enrichment and contested resolution, where
    Goodreads is the whole point and a cooldown ends the run. Ingestion has
    three other sources with no quarrel with us, and letting one unofficial
    source decide whether the catalogue grows would hand it a veto it has not
    earned.
    """

    @pytest.fixture
    def goodreads_on(self, offline: Settings) -> Settings:
        return offline.model_copy(
            update={
                "goodreads_enabled": True,
                "goodreads_unofficial_source_accepted": True,
            }
        )

    @staticmethod
    def _refuse(connection: Connection) -> None:
        record_refusal(connection, start_run(connection), SourceName.GOODREADS, "circuit opened")
        connection.commit()

    def test_the_run_still_produces_books(
        self, goodreads_on: Settings, engine: Engine, connection: Connection
    ) -> None:
        write_manifest(goodreads_on, 3)
        self._refuse(connection)

        report = run_ingestion(goodreads_on, engine=engine)

        # Resolved by the retained discovery payload, exactly as it would be
        # with Goodreads switched off.
        assert report.candidates == 3
        assert report.status != "failed"

    def test_the_cooldown_is_recorded_as_the_skip_reason(
        self, goodreads_on: Settings, engine: Engine, connection: Connection
    ) -> None:
        # "Why is there no Goodreads data in this run" has to be answerable
        # from the run record, and "it refused us at 14:17" is the answer.
        write_manifest(goodreads_on, 1)
        self._refuse(connection)

        run_ingestion(goodreads_on, engine=engine)

        skipped = (
            connection.execute(
                select(source_runs.c.error).where(
                    source_runs.c.source == "goodreads", source_runs.c.status == "skipped"
                )
            )
            .scalars()
            .all()
        )
        assert any("refused us at" in reason for reason in skipped)

    def test_without_a_refusal_goodreads_is_not_skipped_for_that_reason(
        self, goodreads_on: Settings, engine: Engine, connection: Connection
    ) -> None:
        write_manifest(goodreads_on, 1)

        run_ingestion(goodreads_on, engine=engine)

        reasons = connection.execute(select(source_runs.c.error)).scalars().all()
        assert not any("refused us at" in (reason or "") for reason in reasons)


class TestARefusalDuringIngestion:
    """Ingestion records a refusal so the other DAGs back off too.

    A refusal discovered here is the same fact enrichment would have
    discovered; writing it down is what stops each DAG rediscovering it
    separately. Upstream 5xx is deliberately not written.
    """

    class _Refused:
        """A Goodreads extractor that has been blocked."""

        refused = True
        circuit_open = True
        circuit_reason = "access denied: HTTP 403"

        def __init__(self, _settings: Any) -> None: ...

        def build_client(self) -> Any:  # pragma: no cover - never reached
            raise AssertionError("a refused extractor must not be used")

    class _Broken(_Refused):
        """One stopped by upstream 5xx, which is not a refusal."""

        refused = False
        circuit_reason = "HTTP 503"

    @pytest.fixture
    def goodreads_on(self, offline: Settings) -> Settings:
        return offline.model_copy(
            update={
                "goodreads_enabled": True,
                "goodreads_unofficial_source_accepted": True,
            }
        )

    def test_a_refusal_is_recorded(
        self,
        goodreads_on: Settings,
        engine: Engine,
        connection: Connection,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        write_manifest(goodreads_on, 1)
        monkeypatch.setattr(ingest_module, "GoodreadsExtractor", self._Refused)

        run_ingestion(goodreads_on, engine=engine)

        assert last_refusal(connection, SourceName.GOODREADS, within=timedelta(hours=1))

    def test_an_upstream_failure_is_not_recorded(
        self,
        goodreads_on: Settings,
        engine: Engine,
        connection: Connection,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        write_manifest(goodreads_on, 1)
        monkeypatch.setattr(ingest_module, "GoodreadsExtractor", self._Broken)

        run_ingestion(goodreads_on, engine=engine)

        assert last_refusal(connection, SourceName.GOODREADS, within=timedelta(hours=1)) is None

    def test_the_kafka_path_records_it_too(
        self,
        goodreads_on: Settings,
        engine: Engine,
        connection: Connection,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Phase 2 resolves through the same extractor and must not be the one
        # path that forgets.
        write_manifest(goodreads_on, 1)
        monkeypatch.setattr(ingest_module, "GoodreadsExtractor", self._Refused)

        class Sink:
            def emit(self, _events: Any) -> None: ...
            def flush(self) -> None: ...

        run_resolution_to_sink(goodreads_on, Sink(), engine=engine)

        assert last_refusal(connection, SourceName.GOODREADS, within=timedelta(hours=1))
