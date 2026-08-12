"""How the pipeline connects to the catalogue.

Small, and worth pinning: the setting under test is invisible in development
and only shows up as a run that dies twelve minutes in, after the external
calls have already been paid for.
"""

from __future__ import annotations

from pipeline.db import POOL_RECYCLE_SECONDS, build_engine

URL = "postgresql+psycopg://u:p@localhost:5432/catalogue"


class TestConnectionsSurviveAnIdleDatabase:
    def test_connections_are_checked_before_use(self) -> None:
        """The fix for a serverless database suspending mid-run.

        The resolve stage spends minutes in external calls — Goodreads at one
        request per second, Google Books answering 429 until five retries are
        spent — so the connection genuinely idles while the run is alive. Neon
        suspends the compute, and the next write gets a corpse back from the
        pool: "terminating connection due to administrator command".
        """
        engine = build_engine(URL)

        assert engine.pool._pre_ping is True

    def test_connections_are_recycled_before_the_server_gives_up(self) -> None:
        # Below Neon's idle-suspend window, so the pool stops offering a
        # connection the server has already abandoned.
        engine = build_engine(URL)

        assert 0 < engine.pool._recycle <= 300
        assert engine.pool._recycle == POOL_RECYCLE_SECONDS

    def test_callers_can_still_pass_their_own_options(self) -> None:
        engine = build_engine(URL, pool_size=7)

        assert engine.pool.size() == 7
