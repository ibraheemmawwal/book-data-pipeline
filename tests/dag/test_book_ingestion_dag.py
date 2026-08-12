"""The ingestion DAG.

A DAG that fails to import is invisible: Airflow shows a parse error in the UI
and simply never schedules it. These tests are the cheapest way to know that
has not happened, and they run without a database or a scheduler.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

pytestmark = pytest.mark.dag

DAG_ID = "book_ingestion"


class TestItLoads:
    def test_the_dagbag_has_no_import_errors(self, dagbag: Any) -> None:
        # The failure this exists for: a typo makes the DAG vanish from the UI
        # rather than fail loudly.
        assert dagbag.import_errors == {}

    def test_the_dag_is_present(self, dagbag: Any) -> None:
        assert DAG_ID in dagbag.dags

    def test_it_has_no_cycles(self, dagbag: Any) -> None:
        # dag.check_cycle() in Airflow 3; airflow.utils.dag_cycle_tester is
        # deprecated.
        dagbag.dags[DAG_ID].check_cycle()


class TestSchedulingContract:
    def test_it_runs_nightly(self, dagbag: Any) -> None:
        assert dagbag.dags[DAG_ID].schedule == "0 2 * * *"

    def test_catchup_is_off(self, dagbag: Any) -> None:
        # Catching up would replay every missed night against live sources.
        assert dagbag.dags[DAG_ID].catchup is False

    def test_only_one_run_at_a_time(self, dagbag: Any) -> None:
        # Two concurrent runs would resolve the same candidates twice and
        # spend two runs' worth of a per-run source budget.
        assert dagbag.dags[DAG_ID].max_active_runs == 1

    def test_retries_back_off(self, dagbag: Any) -> None:
        args = dagbag.dags[DAG_ID].default_args

        assert args["retries"] == 3
        assert args["retry_exponential_backoff"] is True
        assert args["max_retry_delay"] == timedelta(minutes=30)


class TestTaskGraph:
    def test_the_tasks_match_the_release_architecture(self, dagbag: Any) -> None:
        assert set(dagbag.dags[DAG_ID].task_ids) == {
            "fetch_dump",
            "discover_candidates",
            "resolve_and_load",
            "assess_extraction",
            "finalise_run",
        }

    @pytest.mark.parametrize(
        ("upstream", "downstream"),
        [
            ("discover_candidates", "resolve_and_load"),
            ("discover_candidates", "assess_extraction"),
            ("resolve_and_load", "assess_extraction"),
            ("assess_extraction", "finalise_run"),
        ],
    )
    def test_dependencies_are_wired(self, dagbag: Any, upstream: str, downstream: str) -> None:
        task = dagbag.dags[DAG_ID].get_task(downstream)

        assert upstream in task.upstream_task_ids

    def test_nothing_runs_before_the_fetch(self, dagbag: Any) -> None:
        """Fetching the dump is the root.

        Discovery now depends on it — the pipeline obtains its own input rather
        than assuming a file appeared.
        """
        assert dagbag.dags[DAG_ID].get_task("fetch_dump").upstream_task_ids == set()

    def test_adjudication_is_not_part_of_ingestion(self, dagbag: Any) -> None:
        """It lives in the contested_resolution DAG.

        Running it in both put two writers on the same books and left a stray
        duplicate behind.
        """
        assert "resolve_contested_books" not in dagbag.dags[DAG_ID].task_ids

    def test_discovery_waits_for_the_dump(self, dagbag: Any) -> None:
        upstream = dagbag.dags[DAG_ID].get_task("discover_candidates").upstream_task_ids

        assert upstream == {"fetch_dump"}


class TestTimeouts:
    def test_orchestration_tasks_are_capped_at_an_hour(self, dagbag: Any) -> None:
        assert dagbag.dags[DAG_ID].get_task("discover_candidates").execution_timeout == (
            timedelta(hours=1)
        )

    def test_resolution_gets_longer(self, dagbag: Any) -> None:
        # Goodreads permits one in-flight request and a resolved candidate can
        # need three calls, so a first seed is hours rather than minutes.
        assert dagbag.dags[DAG_ID].get_task("resolve_and_load").execution_timeout == (
            timedelta(hours=6)
        )

    def test_every_task_has_a_timeout(self, dagbag: Any) -> None:
        # A task with no timeout can hold a slot indefinitely.
        dag = dagbag.dags[DAG_ID]

        assert all(dag.get_task(t).execution_timeout is not None for t in dag.task_ids)


class TestXComDiscipline:
    def test_no_task_returns_book_records(self) -> None:
        """XCom is metadata, not a data channel.

        Asserted against the source rather than a run, because the failure is
        someone returning a list of books from a task and it working fine on
        ten records. The names below are what the tasks are allowed to return.
        """
        import inspect

        import book_ingestion_dag

        source = inspect.getsource(book_ingestion_dag)

        for forbidden in ("return clean", "return books", "return records", "return observations"):
            assert forbidden not in source

    def test_discovery_returns_a_path_and_a_count(self) -> None:
        import inspect

        import book_ingestion_dag

        source = inspect.getsource(book_ingestion_dag.book_ingestion)

        assert "manifest_path" in source
        assert "candidates" in source


class TestPhaseTwoShape:
    """The graph Airflow builds when Kafka is enabled.

    The shape is chosen at parse time because that is when Airflow fixes the
    task graph — a run cannot pick a phase. Compose sets the flag on the kafka
    profile and nowhere else, so a default clone always gets phase 1.
    """

    def test_it_still_parses(self, kafka_dagbag: Any) -> None:
        assert kafka_dagbag.import_errors == {}

    def test_resolution_produces_instead_of_loading(self, kafka_dagbag: Any) -> None:
        tasks = set(kafka_dagbag.dags[DAG_ID].task_ids)

        assert "resolve_and_produce" in tasks
        assert "resolve_and_load" not in tasks

    def test_it_emits_a_run_boundary(self, kafka_dagbag: Any) -> None:
        # Without it the consumers would process every event and never learn
        # the run had ended.
        assert "emit_run_boundary" in kafka_dagbag.dags[DAG_ID].task_ids

    def test_the_dag_does_not_load_the_catalogue(self, kafka_dagbag: Any) -> None:
        # Loading belongs to a consumer now. A finalise task here would close a
        # run the consumers are still working on.
        assert "finalise_run" not in kafka_dagbag.dags[DAG_ID].task_ids

    def test_the_boundary_waits_for_the_run_to_be_judged(self, kafka_dagbag: Any) -> None:
        # Closing a topic for a run that resolved nothing would hand the
        # consumers an empty run to finalise.
        boundary = kafka_dagbag.dags[DAG_ID].get_task("emit_run_boundary")

        assert "assess_extraction" in boundary.upstream_task_ids

    def test_resolution_still_gets_the_longer_timeout(self, kafka_dagbag: Any) -> None:
        assert kafka_dagbag.dags[DAG_ID].get_task(
            "resolve_and_produce"
        ).execution_timeout == timedelta(hours=6)

    def test_the_default_shape_is_phase_one(self, dagbag: Any) -> None:
        # A clone with no Kafka configured must not build a graph that produces
        # onto a topic nothing is consuming.
        assert "resolve_and_load" in dagbag.dags[DAG_ID].task_ids
        assert "emit_run_boundary" not in dagbag.dags[DAG_ID].task_ids


class TestTriggerParameters:
    """The form on the trigger page.

    A scheduled run takes every default; an operator triggering by hand can
    narrow the run without editing configuration and restarting the scheduler.
    """

    def test_the_source_can_be_chosen(self, dagbag: Any) -> None:
        param = dagbag.dags[DAG_ID].params.get_param("discovery_source")

        assert param.schema["enum"] == ["openlibrary_dump", "gutendex"]

    def test_the_dump_is_the_default_source(self, dagbag: Any) -> None:
        """It is the only source with coverage of the last century.

        Gutendex is public-domain only, so defaulting to it would silently
        restrict the catalogue to books published before about 1929.
        """
        assert dagbag.dags[DAG_ID].params["discovery_source"] == "openlibrary_dump"

    def test_resuming_is_the_default(self, dagbag: Any) -> None:
        # Otherwise every manual trigger re-reads the dump from the beginning,
        # which is the behaviour the discovery position exists to prevent.
        assert dagbag.dags[DAG_ID].params["resume"] is True

    def test_a_forced_refetch_is_opt_in(self, dagbag: Any) -> None:
        # The cache refreshes itself when the published dump changes; forcing
        # it re-downloads gigabytes for nothing.
        assert dagbag.dags[DAG_ID].params["refresh_dump"] is False

    def test_zero_candidates_means_use_the_configured_default(self, dagbag: Any) -> None:
        assert dagbag.dags[DAG_ID].params["max_candidates"] == 0

    def test_the_candidate_cap_is_bounded(self, dagbag: Any) -> None:
        # A typo on the trigger page should not start a run that reads the
        # whole dump in one pass.
        param = dagbag.dags[DAG_ID].params.get_param("max_candidates")

        assert param.schema["maximum"] <= 100000
