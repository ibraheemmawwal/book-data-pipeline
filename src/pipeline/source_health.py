"""What a source told us last time, remembered across runs.

A circuit breaker lives inside one extractor, in one process, for one run. That
is the right scope for *stopping* — once Goodreads has refused us five times in
a row, the rest of that run has nothing to gain by asking again. It is the
wrong scope for *remembering*, and the two got conflated.

Every Airflow task is a fresh process, so the breaker starts closed each time.
Enrichment runs hourly. A refusal at 14:17 is therefore forgotten by 15:17, and
the next run walks straight back into the same block, collects the same
refusals, trips the same breaker and stops in the same place — which is exactly
the pattern observed across three consecutive runs. Each run behaved correctly
and the sequence behaved like a retry loop.

So the refusal is written down. Any Goodreads path — ingestion, contested
resolution, enrichment — records that the source pushed back, and every path
checks before it opens a client. One run's discovery becomes every run's
restraint, which is the whole point of noticing.

The cooldown is deliberately longer than the schedule interval. A cooldown
shorter than the gap between runs is not a cooldown; it expires before anything
consults it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import structlog
from sqlalchemy import Connection, select
from sqlalchemy.dialects.postgresql import insert

from pipeline.models.db import source_runs
from pipeline.models.domain import SourceName

logger = structlog.get_logger(__name__)

# The status written when a source pushes back, distinct from "skipped" (we
# chose not to ask) and "failed" (we asked and something broke). A source that
# refuses us is a third thing, and the difference matters when reading history:
# skipped is our decision, refused is theirs.
REFUSED = "refused"


class SourceCoolingDownError(Exception):
    """The source refused us recently enough that we should not ask again.

    Not an error in the sense of something going wrong — the run should end
    quietly, the way it does when a source is switched off. It inherits from
    ``Exception`` rather than ``SourceUnavailableError`` because it is raised
    before any client exists, and callers should not confuse "we declined to
    start" with "a request failed".
    """


def record_refusal(connection: Connection, run_id: UUID, source: SourceName, reason: str) -> None:
    """Write down that a source pushed back, so the next run can see it.

    Uses the same ``(run_id, source)`` row the skip path writes, because a run
    that was refused did not also do the source's work — there is nothing to
    overwrite that we would want to keep.
    """
    connection.execute(
        insert(source_runs)
        .values(
            run_id=run_id,
            source=source.value,
            status=REFUSED,
            finished_at=datetime.now(UTC),
            error=reason[:500],
        )
        .on_conflict_do_update(
            index_elements=["run_id", "source"],
            set_={"status": REFUSED, "error": reason[:500], "finished_at": datetime.now(UTC)},
        )
    )
    logger.warning("source.refusal_recorded", source=source.value, reason=reason)


def last_refusal(
    connection: Connection, source: SourceName, *, within: timedelta
) -> datetime | None:
    """When the source last refused us, if that was recent enough to matter.

    ``None`` means clear to proceed. Reads ``finished_at`` rather than
    ``started_at``: the refusal happened at the end of the run that discovered
    it, and dating it from the start would expire the cooldown early by however
    long that run took.
    """
    cutoff = datetime.now(UTC) - within
    return connection.execute(
        select(source_runs.c.finished_at)
        .where(
            source_runs.c.source == source.value,
            source_runs.c.status == REFUSED,
            source_runs.c.finished_at >= cutoff,
        )
        .order_by(source_runs.c.finished_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def ensure_not_cooling_down(
    connection: Connection, source: SourceName, *, cooldown: timedelta
) -> None:
    """Stop the run if the source refused us inside the cooldown window.

    Called before a client is built, so a run in cooldown costs one query and
    sends nothing at all.

    Raises:
        SourceCoolingDownError: the source refused us recently.
    """
    if cooldown <= timedelta(0):
        return

    refused_at = last_refusal(connection, source, within=cooldown)
    if refused_at is None:
        return

    resumes_at = refused_at + cooldown
    remaining = resumes_at - datetime.now(UTC)
    minutes = max(1, int(remaining.total_seconds() // 60))
    logger.info(
        "source.cooling_down",
        source=source.value,
        refused_at=refused_at.isoformat(),
        resumes_at=resumes_at.isoformat(),
    )
    msg = (
        f"{source.value} refused us at {refused_at:%Y-%m-%d %H:%M UTC}; "
        f"waiting another {minutes} min before asking again"
    )
    raise SourceCoolingDownError(msg)
