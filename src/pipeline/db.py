"""How this pipeline connects to the catalogue.

One place, because the connection needs a setting that is easy to omit and
expensive to omit: a serverless database suspends its compute when idle and
terminates the connections it was holding.

That is not a rare edge. The resolve stage spends minutes in external calls —
Goodreads is limited to one request per second and Google Books answers 429
until five retries are spent — so the database connection genuinely sits unused
for long stretches while a run is very much alive. The pool hands the dead
connection back on the next write, and the run dies with
``AdminShutdown: terminating connection due to administrator command``, some
twelve minutes in and after the external work has already been paid for.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine

# Neon's free tier suspends an idle compute after roughly five minutes.
# Recycling below that keeps the pool from offering a connection the server has
# already given up on.
POOL_RECYCLE_SECONDS = 240


def build_engine(url: str, **options: object) -> Engine:
    """An engine that survives the database going away underneath it.

    ``pool_pre_ping`` costs one round trip per checkout and turns a run-ending
    error into a transparent reconnect. Against a local socket that is 0.4ms;
    against a managed database it is the difference between a run that finishes
    and one that does not.
    """
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=POOL_RECYCLE_SECONDS,
        **options,
    )
