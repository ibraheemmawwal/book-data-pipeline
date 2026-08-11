"""The migrations themselves."""

from __future__ import annotations

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect, text

from pipeline.models.db import metadata

pytestmark = pytest.mark.integration


def test_migration_and_metadata_do_not_drift(migrated_engine: Engine) -> None:
    """The single most valuable schema test.

    Metadata and migrations drifting apart is how a schema change passes CI and
    fails in production. Comparing the migrated database against the declared
    metadata catches it at the point the drift is introduced.
    """
    with migrated_engine.connect() as conn:
        differences = compare_metadata(MigrationContext.configure(conn), metadata)

    assert differences == []


def test_pg_trgm_exists_without_manual_sql(migrated_engine: Engine) -> None:
    # The acceptance criterion is a clean clone with no manual database steps.
    with migrated_engine.connect() as conn:
        installed = conn.execute(
            text("SELECT count(*) FROM pg_extension WHERE extname = 'pg_trgm'")
        ).scalar_one()

    assert installed == 1


def test_every_declared_table_exists(migrated_engine: Engine) -> None:
    present = set(inspect(migrated_engine).get_table_names())

    assert set(metadata.tables) <= present


def test_search_vector_is_a_stored_generated_column(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as conn:
        generated = conn.execute(
            text(
                "SELECT is_generated FROM information_schema.columns "
                "WHERE table_name = 'books' AND column_name = 'search_vector'"
            )
        ).scalar_one()

    assert generated == "ALWAYS"


def test_search_vector_weights_title_above_description(migrated_engine: Engine) -> None:
    # The weighting is the ranking contract; a title match must outrank a
    # description match or search returns whatever mentions a word most.
    with migrated_engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO books (identity_key, title, description, content_hash) "
                "VALUES ('fallback:a', 'Whales', 'about ships', 'h1'), "
                "       ('fallback:b', 'Ships', 'about whales', 'h2')"
            )
        )
        ranked = (
            conn.execute(
                text(
                    "SELECT title FROM books, websearch_to_tsquery('english', 'whales') q "
                    "WHERE search_vector @@ q ORDER BY ts_rank(search_vector, q) DESC"
                )
            )
            .scalars()
            .all()
        )
        conn.rollback()

    assert ranked[0] == "Whales"
