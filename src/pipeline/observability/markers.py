"""Durable run-boundary tracking.

A consumer that tracked markers in memory would lose the run boundary across a
restart: it commits offsets past the markers it has seen, resumes after them,
never sees them again, and leaves the run in ``processing`` forever — with
every book correctly loaded, which is what makes that failure so easy to miss.

So observation is a row, and completion is a query. Nothing here depends on how
long a consumer has been alive.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Connection, func, select, update
from sqlalchemy.dialects.postgresql import insert

from pipeline.models.db import ingestion_runs, run_partition_markers, run_topic_partitions

logger = structlog.get_logger(__name__)


def freeze_topology(
    connection: Connection, run_id: UUID, topic: str, expected_partitions: int
) -> None:
    """Record how many partitions this run's markers will cover.

    Written by the barrier before it emits anything. An event must never be the
    authority for topology: a marker carrying its own expectation would let a
    mis-produced event redefine what completion means.
    """
    connection.execute(
        insert(run_topic_partitions)
        .values(run_id=run_id, topic=topic, expected_partitions=expected_partitions)
        .on_conflict_do_nothing(index_elements=["run_id", "topic"])
    )


def record_marker(connection: Connection, run_id: UUID, topic: str, partition: int) -> bool:
    """Note that a boundary marker was observed.

    Called before the offset commit, so a crash in between causes a redelivery
    that re-inserts the same primary key and changes nothing.

    Returns ``False`` when no topology has been frozen for this run and topic.
    The foreign key would refuse the row anyway, and letting that raise would
    kill the consumer on a marker it can never accept, blocking every
    well-formed event behind it. A marker with no expectation to compare
    against cannot complete anything, so it is logged and skipped.
    """
    frozen = connection.execute(
        select(run_topic_partitions.c.expected_partitions).where(
            run_topic_partitions.c.run_id == run_id,
            run_topic_partitions.c.topic == topic,
        )
    ).scalar_one_or_none()

    if frozen is None:
        logger.warning(
            "marker.no_frozen_topology",
            run_id=str(run_id),
            topic=topic,
            partition=partition,
        )
        return False

    connection.execute(
        insert(run_partition_markers)
        .values(run_id=run_id, topic=topic, partition=partition)
        .on_conflict_do_nothing(index_elements=["run_id", "topic", "partition"])
    )
    return True


def is_topic_complete(connection: Connection, run_id: UUID, topic: str) -> bool:
    """Whether every partition of ``topic`` has reported for this run.

    A query rather than accumulated state, which is what makes a mid-run
    restart recover instead of stall.
    """
    expected = connection.execute(
        select(run_topic_partitions.c.expected_partitions).where(
            run_topic_partitions.c.run_id == run_id,
            run_topic_partitions.c.topic == topic,
        )
    ).scalar_one_or_none()

    if expected is None:
        # No frozen topology means the barrier has not run for this topic yet.
        return False

    observed = connection.execute(
        select(func.count())
        .select_from(run_partition_markers)
        .where(
            run_partition_markers.c.run_id == run_id,
            run_partition_markers.c.topic == topic,
        )
    ).scalar_one()

    return bool(observed >= expected)


def runs_awaiting(connection: Connection, topic: str) -> list[UUID]:
    """Runs still in ``processing`` that have a frozen topology for ``topic``.

    The startup sweep reads this. Without it a consumer that restarted after
    seeing some markers would never revisit them, and the run would sit in
    ``processing`` forever.
    """
    rows = connection.execute(
        select(run_topic_partitions.c.run_id)
        .select_from(
            run_topic_partitions.join(
                ingestion_runs, ingestion_runs.c.id == run_topic_partitions.c.run_id
            )
        )
        .where(
            run_topic_partitions.c.topic == topic,
            ingestion_runs.c.status == "processing",
        )
    ).scalars()
    return list(rows)


def mark_processing(
    connection: Connection, run_id: UUID, *, records_extracted: int | None = None
) -> None:
    """Move a run from ``running`` to ``processing``.

    The DAG finishes when extraction does; the consumers carry the run from
    there, and this is the handover — which makes it the only moment anything
    knows how many observations were published. Recording it here is why a
    finished run can say what it produced instead of reporting zero.
    """
    values: dict[str, Any] = {"status": "processing", "extraction_ended_at": func.now()}
    if records_extracted is not None:
        values["records_extracted"] = records_extracted
    connection.execute(
        update(ingestion_runs)
        .where(ingestion_runs.c.id == run_id, ingestion_runs.c.status == "running")
        .values(**values)
    )


def record_loaded(
    connection: Connection, run_id: UUID, *, loaded: int = 0, rejected: int = 0
) -> None:
    """Add to a run's load tally.

    Additive because the consumers are plural: three of them write the same
    run's books from three partitions, and no one of them can state the total.

    Counting only records that *changed* the catalogue is what makes this safe
    under redelivery. A replayed batch loads to ``unchanged`` by construction —
    that is the idempotency guarantee — so it adds nothing here, where a naive
    count of everything processed would inflate the total on every retry.
    """
    if not loaded and not rejected:
        return
    connection.execute(
        update(ingestion_runs)
        .where(ingestion_runs.c.id == run_id)
        .values(
            records_loaded=func.coalesce(ingestion_runs.c.records_loaded, 0) + loaded,
            records_rejected=func.coalesce(ingestion_runs.c.records_rejected, 0) + rejected,
        )
    )


def finalise_if_complete(connection: Connection, run_id: UUID, topic: str) -> bool:
    """Close a run once ``topic`` has fully reported.

    Guarded on ``status = 'processing'`` so a redelivered marker cannot
    re-finalise or re-time a run that is already closed.
    """
    if not is_topic_complete(connection, run_id, topic):
        return False

    result = connection.execute(
        update(ingestion_runs)
        .where(ingestion_runs.c.id == run_id, ingestion_runs.c.status == "processing")
        .values(status="success", processing_ended_at=func.now())
    )
    if result.rowcount:
        logger.info("run.finalised", run_id=str(run_id), topic=topic)
    return bool(result.rowcount)
