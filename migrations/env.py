"""Alembic environment.

The database URL comes from ``Settings`` rather than ``alembic.ini`` so that
migrations, the CLI and the load layer cannot disagree about which database
they are talking to — and so no connection string is ever committed.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from pipeline.config import Settings
from pipeline.models.db import metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate and the drift test both compare against this.
target_metadata = metadata


def _database_url() -> str:
    """Prefer an explicitly supplied URL, else the configured one.

    Tests point this at a throwaway container by setting the option directly.
    """
    return config.get_main_option("sqlalchemy.url") or Settings().database_url  # type: ignore[call-arg]


def run_migrations_offline() -> None:
    """Emit SQL without a live connection, for review or manual application."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
