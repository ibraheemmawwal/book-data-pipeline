"""The adjudication DAG.

Separate from ingestion because it asks a different question over a different
timespan with a different failure mode. These tests pin that separation, since
the temptation to fold it back into the pipeline is exactly what produced a
stray record when both wrote to the same books.
"""

from __future__ import annotations

from typing import Any

import pytest

DAG_ID = "contested_resolution"


class TestGraph:
    def test_it_parses(self, dagbag: Any) -> None:
        assert dagbag.import_errors == {}
        assert DAG_ID in dagbag.dags

    def test_finding_is_separate_from_resolving(self, dagbag: Any) -> None:
        """A run can be inspected before it spends anything.

        The query makes no external request; only the task after it does.
        """
        dag = dagbag.dags[DAG_ID]

        assert dag.get_task("find_contested_books").upstream_task_ids == set()
        assert dag.get_task("resolve_through_goodreads").upstream_task_ids == {
            "find_contested_books"
        }

    def test_the_report_sees_both_halves(self, dagbag: Any) -> None:
        # It has to distinguish "nothing was contested" from "the tie-breaker
        # was switched off", which needs the finding as well as the outcome.
        upstream = dagbag.dags[DAG_ID].get_task("report_resolution").upstream_task_ids

        assert upstream == {"find_contested_books", "resolve_through_goodreads"}

    def test_it_has_no_cycles(self, dagbag: Any) -> None:
        dagbag.dags[DAG_ID].check_cycle()


class TestScheduling:
    def test_it_runs_less_often_than_ingestion(self, dagbag: Any) -> None:
        """Conflicts accumulate at the rate books arrive, not faster.

        Re-examining the same records nightly would ask a restricted source
        the same question seven times a week for one answer.
        """
        contested = dagbag.dags[DAG_ID].schedule
        ingestion = dagbag.dags["book_ingestion"].schedule

        assert str(contested).startswith("0 4 * * ")
        assert str(ingestion) == "0 2 * * *"

    def test_only_one_run_at_a_time(self, dagbag: Any) -> None:
        # Two concurrent runs would re-resolve the same books and spend two
        # runs' worth of a per-run budget.
        assert dagbag.dags[DAG_ID].max_active_runs == 1

    def test_it_does_not_catch_up(self, dagbag: Any) -> None:
        # Replaying missed weeks would query a restricted source repeatedly for
        # a judgement that is only about the catalogue's current state.
        assert dagbag.dags[DAG_ID].catchup is False

    def test_tasks_are_bounded_in_time(self, dagbag: Any) -> None:
        for task in dagbag.dags[DAG_ID].tasks:
            assert task.execution_timeout is not None, task.task_id


class TestSeparationFromIngestion:
    def test_ingestion_no_longer_adjudicates(self, dagbag: Any) -> None:
        """One writer per set of books.

        Running adjudication in both DAGs put two writers on the same records
        and left a stray duplicate behind.
        """
        assert "resolve_contested_books" not in dagbag.dags["book_ingestion"].task_ids

    def test_the_two_dags_share_no_tasks(self, dagbag: Any) -> None:
        ingestion = set(dagbag.dags["book_ingestion"].task_ids)
        contested = set(dagbag.dags[DAG_ID].task_ids)

        assert ingestion.isdisjoint(contested)


class TestFailureIsolation:
    def test_it_retries(self, dagbag: Any) -> None:
        # The tie-breaker can rate-limit; a transient refusal should not need a
        # human to re-trigger the week.
        for task in dagbag.dags[DAG_ID].tasks:
            assert task.retries >= 1, task.task_id

    @pytest.mark.parametrize("tag", ["catalogue", "quality"])
    def test_it_is_tagged_as_a_quality_job(self, dagbag: Any, tag: str) -> None:
        # It corrects the catalogue rather than growing it, and the tag is how
        # someone scanning the UI can tell.
        assert tag in dagbag.dags[DAG_ID].tags
