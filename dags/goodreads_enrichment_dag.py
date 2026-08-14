"""Working through Goodreads records that arrived without their detail.

An export supplies a title, its authors, a rating and a cover. The year, the
ISBN, the page count and the series live on the book's own page, so ten
thousand imported records are ten thousand fetches waiting to happen.

Its own DAG rather than a step in ingestion, for three reasons.

**It is a different question over a different timespan.** Ingestion resolves
candidates never seen before; this revisits records already held. Folding them
together would put one run's failure in charge of the other's budget, and it is
this one that gets blocked.

**It is paced by what the source will bear, not by how fast we could go.** The
backlog is finite and not urgent: a bounded slice each hour reaches the end in
days without ever looking like a crawl. A single run that tried to clear it
would take six hours and be refused somewhere in the middle.

**It must be pausable on its own.** When Goodreads starts answering 202 with an
empty body, the right response is to stop asking — without stopping ingestion,
which does not depend on it.

The block is global, so moving to the next book does not help. Its length
varies: most clear in seconds — measured at five seconds between requests, one
refusal was followed immediately by a success — while a bad one took between
four and five minutes. So a run waits it out in place, escalating ten seconds
to a minute to five, rather than ending over it. Only a block that survives all
three ends the run, and that refusal outlives the run that found it: every Airflow
task is a fresh process, so a breaker that tripped at 14:17 would otherwise be
forgotten by 15:17 and the block rediscovered hourly.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pendulum
from airflow.sdk import Param, dag, task

DAG_ID = "goodreads_enrichment"

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "catalogue",
    # One retry, not two. A retry is for the run that died of something other
    # than the source — a dropped connection, a restarted worker — and the
    # source's own refusals are handled inside the run and remembered across
    # runs, so a second retry buys nothing and costs an interval: 17 minutes of
    # work plus a 10-minute wait plus 17 more already overruns the half hour.
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    # Shorter than the interval it runs in, so a stuck run is killed before the
    # next one is due rather than holding the slot and queueing it.
    "execution_timeout": timedelta(minutes=25),
}


@dag(
    dag_id=DAG_ID,
    # Every half hour, and bounded by what the source will bear rather than by
    # how fast we could go. At one request every thirty seconds — the spacing
    # at which the block stops triggering at all — 25 records is about 17
    # minutes, so each run finishes with a third of its interval spare.
    #
    # Twice as often rather than twice as long, deliberately. Spare time in the
    # interval is what stops a slow run turning into a queued one, and queued
    # intervals are what made failing a run look like the DAG re-triggering
    # itself. Throughput comes from the number of intervals; the slice only
    # decides how much of one gets used.
    schedule="17,47 * * * *",
    start_date=pendulum.datetime(2026, 8, 1, tz="UTC"),
    catchup=False,
    # Two runs would fetch the same head of the queue twice and spend two runs'
    # worth of politeness on one slice.
    max_active_runs=1,
    # Stop after three consecutive failures instead of grinding on.
    #
    # With one run at a time and a slice that takes most of its hour, a
    # scheduled interval is usually waiting behind the active run — so failing
    # a run hands the slot straight to the next one, which looks exactly like
    # the DAG re-triggering itself. It is draining a backlog, not creating one,
    # but the effect is the same: a broken run repeats until someone pauses the
    # DAG by hand, and the only signal is a column of red.
    #
    # Three, not one: a single failure is usually the source having a bad
    # minute, which the next run genuinely does recover from. Three in a row is
    # something that needs a person.
    max_consecutive_failed_dag_runs=3,
    default_args=DEFAULT_ARGS,
    tags=["catalogue", "quality"],
    doc_md=__doc__,
    params={
        "limit": Param(
            None,
            type=["null", "integer"],
            minimum=1,
            maximum=2000,
            title="Records to fetch",
            description=(
                "How many imported records to complete this run. Defaults to "
                "PIPELINE_ENRICH_MAX_PER_RUN. Each record costs one page fetch, "
                "and a second only when the first withholds an ISBN or a year."
            ),
        ),
    },
)
def goodreads_enrichment() -> None:
    """Find imported records, fetch their detail, report what changed."""

    @task
    def count_pending() -> dict[str, Any]:
        """How much of the backlog is left.

        Its own task because it is the number that says whether this DAG should
        still be running at all, and it costs one query and no external
        requests. A run with nothing to do should say so before it builds a
        client.
        """
        from pipeline.config import Settings
        from pipeline.db import build_engine
        from pipeline.enrich import count_unenriched

        settings = Settings()
        pending = count_unenriched(build_engine(settings.database_url))
        return {"pending": pending, "limit": settings.enrich_max_per_run}

    @task
    def fetch_detail(found: dict[str, Any], **context: Any) -> dict[str, Any]:
        """Complete each record from its Goodreads id.

        Bounded four ways: the per-run slice, one request every two seconds
        throughout, a block waited out in place for up to about six minutes, and — if
        it outlasts that — a circuit that ends the run and a cooldown that
        keeps the next one away. None is a performance choice; they are the
        terms on which this source is used at all.
        """
        from pipeline.config import Settings
        from pipeline.enrich import enrich_goodreads
        from pipeline.extract.goodreads import GoodreadsNotAcceptedError
        from pipeline.source_health import SourceCoolingDownError

        if not found["pending"]:
            return {"skipped": "nothing pending", "queried": 0}

        settings = Settings()
        limit = context["params"].get("limit") or settings.enrich_max_per_run

        try:
            report = enrich_goodreads(settings, limit=limit)
        except (GoodreadsNotAcceptedError, SourceCoolingDownError) as error:
            # Neither is a failure. Without the gates the catalogue is exactly
            # what the documented sources described; and a run that declines to
            # ask a source that just refused us has done the right thing, so
            # marking it red would train everyone to ignore red.
            return {"skipped": str(error)[:140], "queried": 0}

        return {
            "pending_before": report.pending,
            "queried": report.queried,
            "enriched": report.enriched,
            "unchanged": report.unchanged,
            "failed": report.failed,
            "loaded": report.loaded,
            "run_id": str(report.run_id) if report.run_id else None,
            "errors": report.errors[:5],
        }

    @task
    def report_progress(found: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
        """Say what happened, and how much is left.

        Separate from the fetching so the numbers survive a retry of neither:
        the work is done and recorded by the time this runs, and its only job
        is to make the result legible.
        """
        import structlog

        remaining = max(0, found["pending"] - outcome.get("enriched", 0))
        structlog.get_logger(__name__).info(
            "dag.enrichment_complete",
            queried=outcome.get("queried", 0),
            enriched=outcome.get("enriched", 0),
            remaining=remaining,
            skipped=outcome.get("skipped"),
        )
        return {**outcome, "remaining": remaining}

    pending = count_pending()
    report_progress(pending, fetch_detail(pending))


goodreads_enrichment()
