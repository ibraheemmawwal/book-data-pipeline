"""Adjudicating books the documented sources disagree about.

Separate from ingestion on purpose, for three reasons.

**A different question.** Ingestion asks what the sources say. This asks which
of them is right where they conflict — a judgement made after the fact, over
the whole catalogue, not over one night's candidates.

**A different cadence.** Conflicts accumulate slowly. Running this every time
books are ingested would re-examine the same records nightly; weekly is closer
to the rate at which the answer changes.

**A different failure.** The tie-breaker uses a restricted source that can
refuse, rate-limit, or be switched off entirely. None of that should mark an
ingestion run as failed, and an ingestion failure should not stop adjudication
of records already in the catalogue.

The source's terms restrict automated collection, so both gates still apply and
every task here is a no-op without them. Scheduling changes when that decision
is checked, not whether it is.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pendulum
from airflow.sdk import dag, task

DAG_ID = "contested_resolution"

DEFAULT_ARGS = {
    "owner": "catalogue",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(minutes=30),
}


@dag(
    dag_id=DAG_ID,
    # Weekly. Conflicts accumulate at the rate books are ingested, and a
    # restricted source should be asked as rarely as the question allows.
    schedule="0 4 * * 0",
    start_date=pendulum.datetime(2026, 8, 1, tz="UTC"),
    catchup=False,
    # Two concurrent runs would re-resolve the same books and spend two runs'
    # worth of a per-run budget. It is also the race that produced a stray
    # record when ingestion and adjudication overlapped.
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["catalogue", "quality"],
    doc_md=__doc__,
)
def contested_resolution() -> None:
    """Find contested books, adjudicate them, report what changed."""

    @task
    def find_contested_books() -> dict[str, Any]:
        """Identify the books whose sources conflict.

        Makes no external request. Separating the query from the resolution
        means a run can be inspected before it spends anything — and when the
        set is empty, which is the normal state of a healthy catalogue, nothing
        downstream needs to happen.
        """

        from pipeline.config import Settings
        from pipeline.contested import find_contested
        from pipeline.db import build_engine

        settings = Settings()
        books = find_contested(
            build_engine(settings.database_url),
            minimum_conflicts=settings.contested_min_conflicts,
            limit=settings.contested_max_per_run,
        )

        return {
            "count": len(books),
            "minimum_conflicts": settings.contested_min_conflicts,
            "limit": settings.contested_max_per_run,
            # A sample, not the set: the whole list would put catalogue content
            # into Airflow's metadata database, and the resolution task reads
            # from PostgreSQL anyway.
            "worst": [
                {"title": b["title"][:80], "conflicts": b["conflicts"], "sources": b["sources"]}
                for b in books[:5]
            ],
        }

    @task
    def resolve_through_goodreads(found: dict[str, Any]) -> dict[str, Any]:
        """Ask the tie-breaker about each contested book.

        Bounded three ways: only books above the conflict threshold, never more
        than the per-run limit, and the circuit breaker stops the whole run on
        the first refusal. Re-probing a source that has pushed back is what the
        containment rules forbid, and a targeted run is not an exception.
        """
        from pipeline.config import Settings
        from pipeline.contested import resolve_contested
        from pipeline.extract.goodreads import GoodreadsNotAcceptedError

        if not found["count"]:
            return {"skipped": "no contested books", "queried": 0}

        settings = Settings()
        if not settings.resolve_contested_enabled:
            return {"skipped": "disabled by configuration", "queried": 0}

        try:
            report = resolve_contested(
                settings,
                minimum_conflicts=settings.contested_min_conflicts,
                limit=settings.contested_max_per_run,
            )
        except GoodreadsNotAcceptedError as error:
            # Not a failure. The tie-breaker is optional, and without it the
            # catalogue is exactly what the documented sources described.
            return {"skipped": str(error)[:140], "queried": 0}

        return {
            "contested": report.contested,
            "queried": report.queried,
            "resolved": report.resolved,
            "loaded": report.loaded,
            "unresolved": report.unresolved,
            "errors": report.errors[:5],
        }

    @task
    def report_resolution(found: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
        """Say what the run did, including when it did nothing.

        A skipped run and a run that found nothing look identical from the
        outside and mean opposite things — one is a healthy catalogue, the
        other is a tie-breaker that has been switched off and forgotten.
        """
        import structlog

        logger = structlog.get_logger(__name__)

        if outcome.get("skipped"):
            status = "skipped"
        elif not found["count"]:
            status = "nothing_contested"
        elif outcome.get("resolved"):
            status = "resolved"
        else:
            status = "no_answers"

        summary = {
            "status": status,
            "contested_found": found["count"],
            "queried": outcome.get("queried", 0),
            "resolved": outcome.get("resolved", 0),
            "loaded": outcome.get("loaded", 0),
            "reason": outcome.get("skipped"),
        }
        logger.info("contested_resolution.complete", **summary)
        return summary

    found = find_contested_books()
    outcome = resolve_through_goodreads(found)
    report_resolution(found, outcome)


contested_resolution()
