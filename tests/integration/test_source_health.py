"""Refusals that outlive the run that discovered them.

The circuit breaker lives in one process and every scheduled task is a fresh
process, so on its own it stops a run and teaches the next one nothing. These
tests pin the part that closes that gap: a refusal is written down, and every
Goodreads path consults it before opening a client.

Against the real schema, because the whole mechanism is a row and a query — a
mock would be testing the test.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import Connection, select

from pipeline.models.db import source_runs
from pipeline.models.domain import SourceName
from pipeline.observability.runs import record_source_skip, start_run
from pipeline.source_health import (
    REFUSED,
    SourceCoolingDownError,
    ensure_not_cooling_down,
    last_refusal,
    record_refusal,
)

pytestmark = pytest.mark.integration

HOUR = timedelta(hours=1)


def _refuse(connection: Connection, *, ago: timedelta = timedelta(0)) -> UUID:
    """Record a refusal, optionally backdated."""
    run_id = start_run(connection)
    record_refusal(connection, run_id, SourceName.GOODREADS, "circuit opened")
    if ago:
        connection.execute(
            source_runs.update()
            .where(source_runs.c.run_id == run_id)
            .values(finished_at=datetime.now(UTC) - ago)
        )
    return run_id


class TestRememberingARefusal:
    def test_a_fresh_database_is_clear_to_proceed(self, connection: Connection) -> None:
        assert last_refusal(connection, SourceName.GOODREADS, within=HOUR) is None
        ensure_not_cooling_down(connection, SourceName.GOODREADS, cooldown=HOUR)

    def test_a_refusal_stops_the_next_run(self, connection: Connection) -> None:
        _refuse(connection)

        with pytest.raises(SourceCoolingDownError) as caught:
            ensure_not_cooling_down(connection, SourceName.GOODREADS, cooldown=HOUR)

        # The message has to carry when it happened and how long is left, or
        # the operator's only recourse is to read the table by hand.
        assert "refused us at" in str(caught.value)
        assert "min before asking again" in str(caught.value)

    def test_the_cooldown_expires(self, connection: Connection) -> None:
        _refuse(connection, ago=timedelta(hours=2))

        assert last_refusal(connection, SourceName.GOODREADS, within=HOUR) is None
        ensure_not_cooling_down(connection, SourceName.GOODREADS, cooldown=HOUR)

    def test_the_boundary_belongs_to_the_cooldown(self, connection: Connection) -> None:
        # Just inside the window still counts; the alternative is a run that
        # slips through in the second before expiry.
        _refuse(connection, ago=HOUR - timedelta(minutes=1))

        with pytest.raises(SourceCoolingDownError):
            ensure_not_cooling_down(connection, SourceName.GOODREADS, cooldown=HOUR)

    def test_the_most_recent_refusal_wins(self, connection: Connection) -> None:
        _refuse(connection, ago=timedelta(hours=5))
        _refuse(connection, ago=timedelta(minutes=10))

        found = last_refusal(connection, SourceName.GOODREADS, within=HOUR)

        assert found is not None
        # Not the older one, which would have expired and reopened the tap.
        assert datetime.now(UTC) - found < timedelta(minutes=30)


class TestWhatDoesNotCount:
    def test_a_refusal_is_per_source(self, connection: Connection) -> None:
        _refuse(connection)

        # Open Library has no quarrel with us and must not inherit one.
        ensure_not_cooling_down(connection, SourceName.OPENLIBRARY, cooldown=HOUR)

    def test_our_own_skip_is_not_their_refusal(self, connection: Connection) -> None:
        # "We chose not to ask" and "they told us no" are different facts, and
        # conflating them would make a disabled source look like a blocked one.
        run_id = start_run(connection)
        record_source_skip(connection, run_id, SourceName.GOODREADS, "disabled")

        assert last_refusal(connection, SourceName.GOODREADS, within=HOUR) is None

    def test_a_zero_cooldown_disables_the_wait(self, connection: Connection) -> None:
        _refuse(connection)

        ensure_not_cooling_down(connection, SourceName.GOODREADS, cooldown=timedelta(0))


class TestTheRecord:
    def test_it_is_distinguishable_from_a_skip(self, connection: Connection) -> None:
        run_id = _refuse(connection)

        row = connection.execute(
            select(source_runs.c.status, source_runs.c.error, source_runs.c.finished_at).where(
                source_runs.c.run_id == run_id
            )
        ).one()

        assert row.status == REFUSED
        assert row.error == "circuit opened"
        assert row.finished_at is not None

    def test_a_second_refusal_in_one_run_does_not_duplicate(self, connection: Connection) -> None:
        # (run_id, source) is unique; a retry inside one run must update rather
        # than raise, or the recovery path becomes the thing that fails.
        run_id = start_run(connection)
        record_refusal(connection, run_id, SourceName.GOODREADS, "circuit opened")
        record_refusal(connection, run_id, SourceName.GOODREADS, "circuit opened again")

        rows = connection.execute(
            select(source_runs.c.error).where(source_runs.c.run_id == run_id)
        ).all()

        assert len(rows) == 1
        assert rows[0].error == "circuit opened again"

    def test_it_upgrades_a_skip_recorded_earlier_in_the_same_run(
        self, connection: Connection
    ) -> None:
        run_id = start_run(connection)
        record_source_skip(connection, run_id, SourceName.GOODREADS, "planned skip")
        record_refusal(connection, run_id, SourceName.GOODREADS, "circuit opened")

        assert last_refusal(connection, SourceName.GOODREADS, within=HOUR) is not None

    def test_a_long_reason_is_truncated_not_rejected(self, connection: Connection) -> None:
        run_id = start_run(connection)
        record_refusal(connection, run_id, SourceName.GOODREADS, "x" * 5000)

        error = connection.execute(
            select(source_runs.c.error).where(source_runs.c.run_id == run_id)
        ).scalar_one()

        assert len(error) == 500
