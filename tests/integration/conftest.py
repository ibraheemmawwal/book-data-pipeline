"""Integration fixtures: a real PostgreSQL container, migrated by Alembic.

Mocks cannot prove any of what this milestone claims. Idempotency depends on
ON CONFLICT semantics, the merge depends on cascade ordering, and the schema
depends on a generated tsvector column — none of which exist outside a real
PostgreSQL.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine, text
from testcontainers.community.postgres import PostgresContainer

from pipeline.models.db import metadata

REPO_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """A throwaway PostgreSQL 16, matching the deployment target."""
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as container:
        yield container.get_connection_url()


@pytest.fixture(scope="session")
def migrated_engine(postgres_url: str) -> Iterator[Engine]:
    """An engine against a database migrated with the real Alembic scripts.

    Running the actual migrations rather than ``metadata.create_all`` is the
    point: it proves the migrations work, which is what production will run.
    """
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", postgres_url)
    command.upgrade(config, "head")

    engine = create_engine(postgres_url)
    yield engine
    engine.dispose()


@pytest.fixture
def connection(migrated_engine: Engine) -> Iterator[Connection]:
    """A clean connection with every table truncated.

    Truncating between tests keeps them independent without paying for a new
    container each time.
    """
    with migrated_engine.connect() as conn:
        conn.execute(
            text(
                # Derived from metadata rather than a hand-written list: a
                # table added to the schema and forgotten here leaks state
                # between tests and produces failures that only appear in a
                # full run.
                f"TRUNCATE {', '.join(sorted(metadata.tables))} RESTART IDENTITY CASCADE"
            )
        )
        conn.commit()
        yield conn


@pytest.fixture
def engine(migrated_engine: Engine, connection: Connection) -> Engine:
    """The engine the loader writes through.

    Depends on ``connection`` so the truncate runs first; the loader opens its
    own connections because it owns the batch transaction boundary.
    """
    _ = connection
    return migrated_engine
