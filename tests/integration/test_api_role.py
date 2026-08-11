"""The read-only role the catalogue API connects as.

Least privilege is only worth claiming if it is enforced, and the failure mode
of getting this wrong is silent: the API keeps working, with more power than it
should have, until something writes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import ProgrammingError

pytestmark = pytest.mark.integration

GRANTS = Path(__file__).parent.parent.parent / "scripts" / "grant_api_role.sql"


@pytest.fixture
def api_engine(migrated_engine: Engine) -> Engine:
    """An engine connected as the API's read-only role."""
    # Strip psql-only directives and comments before splitting. Left in,
    # a comment block becomes part of the statement that follows it, and
    # skipping chunks that "start with --" then silently drops the statement
    # they document — which is how ALTER DEFAULT PRIVILEGES went missing.
    raw = GRANTS.read_text().replace(':"api_role"', "catalogue_api")
    sql = "\n".join(
        line for line in raw.splitlines() if not line.strip().startswith(("--", "\\set"))
    )

    with migrated_engine.begin() as conn:
        # DROP OWNED fails if the role was never created, so create first and
        # make the whole setup re-runnable.
        conn.execute(
            text(
                "DO $$ BEGIN "
                "IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'catalogue_api') "
                "THEN CREATE ROLE catalogue_api LOGIN PASSWORD 'readonly'; END IF; "
                "END $$"
            )
        )
    with migrated_engine.begin() as conn:
        for statement in filter(None, (part.strip() for part in sql.split(";"))):
            conn.execute(text(statement))

    url = migrated_engine.url.set(username="catalogue_api", password="readonly")
    return create_engine(url)


def test_the_api_role_can_read(api_engine: Engine) -> None:
    with api_engine.connect() as conn:
        conn.execute(text("SELECT count(*) FROM books")).scalar_one()


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO books (identity_key, title, content_hash) VALUES ('x', 'y', 'z')",
        "UPDATE books SET title = 'nope'",
        "DELETE FROM books",
        "CREATE TABLE sneaky (id int)",
    ],
)
def test_the_api_role_cannot_write(api_engine: Engine, statement: str) -> None:
    with api_engine.connect() as conn, pytest.raises(ProgrammingError, match="permission denied"):
        conn.execute(text(statement))


def test_default_privileges_cover_a_table_added_later(
    api_engine: Engine, migrated_engine: Engine
) -> None:
    # The case the ALTER DEFAULT PRIVILEGES line exists for: a table created by
    # a future migration must be readable without re-running the grants.
    with migrated_engine.begin() as conn:
        conn.execute(text("CREATE TABLE added_later (id int)"))

    try:
        with api_engine.connect() as conn:
            conn.execute(text("SELECT * FROM added_later")).all()
    finally:
        with migrated_engine.begin() as conn:
            conn.execute(text("DROP TABLE added_later"))
