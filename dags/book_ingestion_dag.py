"""The nightly catalogue ingestion DAG.

Airflow's job here is orchestration and accounting, not business logic. Every
task is a thin call into ``pipeline``: the DAG decides ordering, retries and
what a failure means, and the package decides what a book is. That split is why
the whole pipeline can be tested without Airflow installed, and why this file
is short.

Two rules shape the task boundaries.

**XCom carries metadata, never records.** Paths, counts and typed statuses
travel between tasks; books do not. XCom lives in the metadata database, and a
run pushing 6,000 books through it would be writing the catalogue twice, into
the wrong database, in a format nothing can query.

**The graph has two shapes.** In phase 1 the DAG loads the catalogue itself. In
phase 2 it resolves candidates onto ``books.raw``, emits the run boundary, and
finishes — transform and load become long-running consumers. That is a
deliberate narrowing of Airflow's responsibility, so a slow load stops blocking
ingestion instead of holding a task open for hours. Which shape is built is
decided when this file is parsed, because that is when Airflow fixes the graph.

**One source failing must not fail the run.** Provider outages are expected —
Goodreads is an unofficial contract and the documented APIs have quotas — so
resolution records per-source outcomes and returns them. ``assess_extraction``
is the single place that decides whether what came back is good enough, and the
only task that can fail the run for a data reason.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pendulum
from airflow.sdk import dag, task

DAG_ID = "book_ingestion"


# Which shape the graph takes is a parse-time decision, not a runtime one:
# Airflow builds the task graph when it reads this file, so the phases cannot
# be chosen per run. Compose sets this on the kafka profile and nowhere else.
def _kafka_mode() -> bool:
    """Whether to build the phase 2 graph.

    Read through Settings rather than straight from the environment: Settings
    rejects unknown PIPELINE_* variables, so a bare os.environ lookup would
    make every task fail on a variable the DAG itself introduced.
    """
    from pipeline.config import Settings

    try:
        return Settings().kafka_enabled
    except Exception:
        return False


KAFKA_MODE = _kafka_mode()

# The default applies to finite orchestration tasks. Resolution overrides it:
# Goodreads permits one in-flight request and a resolved candidate can need
# three calls, so a first seed of several thousand candidates is hours of work.
DEFAULT_EXECUTION_TIMEOUT = timedelta(hours=1)
RESOLVE_EXECUTION_TIMEOUT = timedelta(hours=6)

DEFAULT_ARGS: dict[str, Any] = {
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": DEFAULT_EXECUTION_TIMEOUT,
}


@dag(
    dag_id=DAG_ID,
    schedule="0 2 * * *",
    start_date=pendulum.datetime(2026, 8, 1, tz="UTC"),
    # Catching up would replay every missed night against live sources for no
    # benefit: the catalogue is a current snapshot, not a time series.
    catchup=False,
    # Two concurrent runs would resolve the same candidates twice and spend two
    # runs' worth of a per-run source budget.
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["catalogue", "etl"],
    doc_md=__doc__,
)
def book_ingestion() -> None:
    """Discover candidates, resolve them, load the catalogue."""

    @task
    def discover_candidates() -> dict[str, Any]:
        """Build the candidate manifest from the pinned dump.

        Returns the manifest path and a count — never candidates. The manifest
        is the deterministic artefact; passing its contents through XCom would
        put the dump in the metadata database.
        """
        from pipeline.config import Settings
        from pipeline.discover import build_manifest

        settings = Settings()
        written = build_manifest(
            settings.openlibrary_dump_path,  # type: ignore[arg-type]
            settings.discovery_manifest_path,
            languages=settings.discovery_language_set(),
            max_candidates=settings.discovery_max_candidates,
            expected_sha256=settings.openlibrary_dump_sha256,
        )
        return {
            "manifest_path": str(settings.discovery_manifest_path),
            "candidates": written,
            "status": "success" if written else "failed",
        }

    @task(execution_timeout=RESOLVE_EXECUTION_TIMEOUT)
    def resolve_and_load(discovery: dict[str, Any]) -> dict[str, Any]:
        """Resolve every candidate and load what came back.

        Returns aggregate counts. A provider failing is recorded here and
        judged by the next task rather than raised: a Goodreads outage with
        healthy fallbacks is not a failed run.
        """
        from pipeline.config import Settings
        from pipeline.ingest import run_ingestion

        report = run_ingestion(Settings())
        return {
            "candidates": report.candidates,
            "resolved": report.resolved,
            "unresolved": report.unresolved,
            "books_inserted": report.books_inserted,
            "books_updated": report.books_updated,
            "books_unchanged": report.books_unchanged,
            "rejected": report.rejected,
            "status": report.status,
            "discovered": discovery["candidates"],
        }

    @task(execution_timeout=RESOLVE_EXECUTION_TIMEOUT)
    def resolve_and_produce(discovery: dict[str, Any]) -> dict[str, Any]:
        """Resolve candidates onto books.raw and stop there.

        Phase 2's extract stage. Canonicalisation and loading belong to the
        consumers now, so this task finishes when the events are on the topic
        rather than when the catalogue is written.
        """
        from pipeline.config import Settings
        from pipeline.ingest import run_resolution_to_sink
        from pipeline.messaging.kafka import KafkaSink

        settings = Settings()
        sink = KafkaSink(settings.kafka_bootstrap_servers, settings.kafka_raw_topic)
        report = run_resolution_to_sink(settings, sink)
        return {
            "candidates": report.candidates,
            "resolved": report.resolved,
            "unresolved": report.unresolved,
            "observations": report.observations,
            "rejected": report.rejected,
            "books_inserted": 0,
            "books_updated": 0,
            "books_unchanged": 0,
            "status": report.status,
            "run_id": str(report.run_id),
            "discovered": discovery["candidates"],
        }

    @task
    def emit_run_boundary(outcome: dict[str, Any]) -> dict[str, Any]:
        """Freeze the topology and write one marker per raw partition.

        The handover. After this the DAG is done and the consumers carry the
        run; without it they would process every event and never learn the run
        had ended.
        """
        from uuid import UUID

        from pipeline.config import Settings
        from pipeline.services import emit_run_boundary as emit

        partitions = emit(Settings(), UUID(outcome["run_id"]))
        return {**outcome, "partitions": partitions}

    @task
    def assess_extraction(discovery: dict[str, Any], outcome: dict[str, Any]) -> str:
        """Decide whether the run produced a usable catalogue.

        The only task allowed to fail the run for a data reason. Discovery
        finding nothing, or no candidate resolving from any source, is a failed
        run. Anything else is success or partial success, and that distinction
        is worth keeping: with a hierarchy of fallible sources, partial is the
        common case rather than an anomaly.
        """
        from airflow.exceptions import AirflowFailException

        if discovery["status"] != "success" or not discovery["candidates"]:
            msg = "discovery produced no candidates; nothing could be resolved"
            raise AirflowFailException(msg)

        if not outcome["resolved"]:
            msg = f"no candidate resolved from any source ({outcome['candidates']} attempted)"
            raise AirflowFailException(msg)

        return str(outcome["status"])

    @task
    def finalise_run(status: str, outcome: dict[str, Any]) -> dict[str, Any]:
        """Publish the run summary.

        ``run_ingestion`` already closed the database record; this is the
        operator-facing view and what appears in the task log.
        """
        import structlog

        structlog.get_logger(__name__).info(
            "dag.run_complete",
            status=status,
            candidates=outcome["candidates"],
            resolved=outcome["resolved"],
            books_inserted=outcome["books_inserted"],
            books_unchanged=outcome["books_unchanged"],
            rejected=outcome["rejected"],
        )
        return {"status": status, **outcome}

    discovery = discover_candidates()

    if KAFKA_MODE:
        outcome = resolve_and_produce(discovery)
        status = assess_extraction(discovery, outcome)
        # The boundary is emitted only after the run is judged worth
        # continuing: closing a topic for a run that resolved nothing would
        # hand the consumers an empty run to finalise.
        emit_run_boundary(outcome).set_upstream(status)
    else:
        outcome = resolve_and_load(discovery)
        status = assess_extraction(discovery, outcome)
        finalise_run(status, outcome)


book_ingestion()
