"""The enrichment DAG's shape.

It exists separately from ingestion because it asks a different question over a
different timespan, and because it must be pausable on its own — when the
source starts refusing us, the right response is to stop asking without
stopping ingestion, which does not depend on it.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

DAG_ID = "goodreads_enrichment"

pytestmark = pytest.mark.dag


class TestTaskGraph:
    def test_each_step_is_its_own_task(self, dagbag: Any) -> None:
        # Counting, fetching and reporting fail differently and are worth
        # seeing apart: an empty backlog is not a refused source.
        assert set(dagbag.dags[DAG_ID].task_ids) == {
            "count_pending",
            "fetch_detail",
            "report_progress",
        }

    def test_fetching_waits_for_the_count(self, dagbag: Any) -> None:
        """A run with nothing to do should not build a client.

        The count costs one query and no external request, which is what makes
        it worth asking first.
        """
        upstream = dagbag.dags[DAG_ID].get_task("fetch_detail").upstream_task_ids

        assert upstream == {"count_pending"}

    def test_the_report_sees_both(self, dagbag: Any) -> None:
        upstream = dagbag.dags[DAG_ID].get_task("report_progress").upstream_task_ids

        assert upstream == {"count_pending", "fetch_detail"}


class TestPacing:
    def test_only_one_run_at_a_time(self, dagbag: Any) -> None:
        # Two runs would fetch the same head of the queue twice and spend two
        # runs' worth of politeness on one slice.
        assert dagbag.dags[DAG_ID].max_active_runs == 1

    def test_it_does_not_catch_up(self, dagbag: Any) -> None:
        """Missed intervals must not become a burst.

        Catchup on an hourly schedule after a day paused would fire twenty-four
        runs at a source that had just started answering again.
        """
        assert dagbag.dags[DAG_ID].catchup is False

    def test_a_run_is_capped_below_its_interval(self, dagbag: Any) -> None:
        """The timeout has to be shorter than the gap between runs.

        A run that outlives its interval holds the only slot, so the next
        interval queues behind it instead of starting — which is how a slow run
        turns into a permanent backlog of scheduled runs.
        """
        timeout = dagbag.dags[DAG_ID].get_task("fetch_detail").execution_timeout

        assert timeout == timedelta(minutes=50)
        assert timeout < timedelta(hours=1)

    def test_the_slice_is_overridable_from_the_trigger_page(self, dagbag: Any) -> None:
        # Working through a backlog by hand wants a different size from the
        # hourly schedule, and neither should need a redeploy.
        assert "limit" in dagbag.dags[DAG_ID].params
