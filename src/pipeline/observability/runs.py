"""Run records.

Every ingestion writes a row before it does any work and closes it afterwards,
so a crashed run is distinguishable from one that never started. That is the
difference between "the catalogue is missing books" and "we know exactly which
run stopped and where".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import Connection, func, update
from sqlalchemy.dialects.postgresql import insert

from pipeline.models.db import ingestion_runs, source_runs
from pipeline.models.domain import SourceName

# Longer than any real run, so a sweep can never close a live one. The longest
# observed is a 5000-candidate ingestion at roughly an hour.
STALE_RUN_AGE = timedelta(hours=12)


def abandon_stale_runs(connection: Connection, *, older_than: timedelta = STALE_RUN_AGE) -> int:
    """Close runs left ``running`` by a kill, and say how many.

    A run that is killed rather than raising — a container restart, a task
    adopted after a scheduler bounce, an OOM — never reaches its own exception
    handler, so its row stays ``running`` for good. Five had accumulated in the
    live catalogue, two of them more than a day old, each claiming work was in
    progress when nothing was.

    Age is the test, and it has to be: the tempting version closes every
    ``running`` row on the grounds that ``max_active_runs=1`` means only one
    can be live. That is per *DAG*. Contested resolution opens runs too, so a
    tie-breaking run starting would have marked a live ingestion failed —
    turning a cosmetic leak into a lie about the run that mattered.

    Runs that reached the consumers are ``processing``, not ``running``, and
    are left alone: those genuinely are still in flight.
    """
    cutoff = datetime.now(UTC) - older_than
    result = connection.execute(
        update(ingestion_runs)
        .where(ingestion_runs.c.status == "running", ingestion_runs.c.started_at < cutoff)
        .values(status="failed", extraction_ended_at=func.now(), processing_ended_at=func.now())
    )
    return int(result.rowcount or 0)


def start_run(connection: Connection, dag_run_id: str | None = None) -> UUID:
    """Open a run and return its id.

    ``dag_run_id`` is unique, so a CLI run gets a ``cli:<uuid4>`` label rather
    than colliding with Airflow's.

    Any run left open by a previous kill is closed first, so "running" means
    running.
    """
    abandon_stale_runs(connection)
    run_id = uuid4()
    connection.execute(
        insert(ingestion_runs).values(
            id=run_id, dag_run_id=dag_run_id or f"cli:{run_id}", status="running"
        )
    )
    return run_id


def record_source_skip(
    connection: Connection, run_id: UUID, source: SourceName, reason: str
) -> None:
    """Note that a source will not run, and why.

    A skipped source has to be visible in the run record; inferring it from a
    gap in the data is how a misconfiguration survives for weeks.
    """
    connection.execute(
        insert(source_runs)
        .values(
            run_id=run_id,
            source=source.value,
            status="skipped",
            finished_at=datetime.now(UTC),
            error=reason,
        )
        .on_conflict_do_update(
            index_elements=["run_id", "source"],
            set_={"status": "skipped", "error": reason},
        )
    )


def finalise_run(  # noqa: PLR0913
    connection: Connection,
    run_id: UUID,
    *,
    status: str,
    records_extracted: int = 0,
    records_loaded: int = 0,
    records_rejected: int = 0,
) -> None:
    """Close a run with its terminal status and counts."""
    connection.execute(
        update(ingestion_runs)
        .where(ingestion_runs.c.id == run_id)
        .values(
            status=status,
            processing_ended_at=func.now(),
            extraction_ended_at=func.now(),
            records_extracted=records_extracted,
            records_loaded=records_loaded,
            records_rejected=records_rejected,
        )
    )
