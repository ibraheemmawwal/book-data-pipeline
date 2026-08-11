"""Create the catalogue schema.

Everything the database needs is here, including the pg_trgm extension: the
acceptance criterion is that a clean clone runs with no manual SQL, and a
migration that assumes an extension already exists fails that on a fresh
database. The extension must also precede the trigram indexes that use it.

Revision ID: 37d934c230b4
Revises:
Create Date: 2026-08-11 10:54:45.096374

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "37d934c230b4"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Before any index using gin_trgm_ops.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "authors",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("normalized_name", sa.Text(), nullable=False),
        sa.Column("birth_year", sa.SmallInteger(), nullable=True),
        sa.Column("death_year", sa.SmallInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "birth_year IS NULL OR birth_year BETWEEN -3000 AND 2100",
            name=op.f("ck_authors_birth_year_range"),
        ),
        sa.CheckConstraint(
            "birth_year IS NULL OR death_year IS NULL OR death_year >= birth_year",
            name=op.f("ck_authors_lifespan_order"),
        ),
        sa.CheckConstraint(
            "death_year IS NULL OR death_year BETWEEN -3000 AND 2100",
            name=op.f("ck_authors_death_year_range"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_authors")),
        sa.UniqueConstraint("normalized_name", name=op.f("uq_authors_normalized_name")),
    )
    op.create_index(
        "idx_authors_name_trgm",
        "authors",
        ["name"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"name": "gin_trgm_ops"},
    )
    op.create_table(
        "books",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("identity_key", sa.Text(), nullable=False),
        sa.Column("isbn13", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("subtitle", sa.Text(), nullable=True),
        sa.Column("published_year", sa.SmallInteger(), nullable=True),
        sa.Column("publisher", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("download_count", sa.BigInteger(), nullable=True),
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("cover_url", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "setweight(to_tsvector('english', coalesce(title, '')), 'A') || setweight(to_tsvector('english', coalesce(subtitle, '')), 'B') || setweight(to_tsvector('english', coalesce(description, '')), 'C')",
                persisted=True,
            ),
            nullable=True,
        ),
        sa.CheckConstraint(
            "isbn13 IS NULL OR isbn13 ~ '^[0-9]{13}$'", name=op.f("ck_books_isbn13_format")
        ),
        sa.CheckConstraint(
            "language IS NULL OR language ~ '^[a-z]{3}$'", name=op.f("ck_books_language_format")
        ),
        sa.CheckConstraint(
            "download_count IS NULL OR download_count >= 0",
            name=op.f("ck_books_download_count_non_negative"),
        ),
        sa.CheckConstraint(
            "page_count IS NULL OR page_count > 0", name=op.f("ck_books_page_count_positive")
        ),
        sa.CheckConstraint(
            "published_year IS NULL OR published_year BETWEEN 1400 AND 2100",
            name=op.f("ck_books_published_year_range"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_books")),
        sa.UniqueConstraint("identity_key", name=op.f("uq_books_identity_key")),
        sa.UniqueConstraint("isbn13", name=op.f("uq_books_isbn13")),
    )
    op.create_index(
        "idx_books_search", "books", ["search_vector"], unique=False, postgresql_using="gin"
    )
    op.create_index(
        "idx_books_title_keyset", "books", [sa.literal_column("lower(title)"), "id"], unique=False
    )
    op.create_index(
        "idx_books_title_trgm",
        "books",
        ["title"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"title": "gin_trgm_ops"},
    )
    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("dag_run_id", sa.Text(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("extraction_ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), server_default=sa.text("'running'"), nullable=False),
        sa.Column("records_extracted", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("records_loaded", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("records_rejected", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.CheckConstraint(
            "status IN ('running', 'processing', 'success', 'partial_success', 'failed')",
            name=op.f("ck_ingestion_runs_status_known"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_runs")),
        sa.UniqueConstraint("dag_run_id", name=op.f("uq_ingestion_runs_dag_run_id")),
    )
    op.create_table(
        "subjects",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("normalized_name", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_subjects")),
        sa.UniqueConstraint("normalized_name", name=op.f("uq_subjects_normalized_name")),
    )
    op.create_table(
        "author_sources",
        sa.Column("author_id", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_author_id", sa.Text(), nullable=False),
        sa.Column("source_birth_year", sa.SmallInteger(), nullable=True),
        sa.Column("source_death_year", sa.SmallInteger(), nullable=True),
        sa.CheckConstraint(
            "source IN ('gutendex', 'openlibrary', 'googlebooks')",
            name=op.f("ck_author_sources_source_known"),
        ),
        sa.CheckConstraint(
            "source_birth_year IS NULL OR source_birth_year BETWEEN -3000 AND 2100",
            name=op.f("ck_author_sources_birth_year_range"),
        ),
        sa.CheckConstraint(
            "source_birth_year IS NULL OR source_death_year IS NULL OR source_death_year >= source_birth_year",
            name=op.f("ck_author_sources_lifespan_order"),
        ),
        sa.CheckConstraint(
            "source_death_year IS NULL OR source_death_year BETWEEN -3000 AND 2100",
            name=op.f("ck_author_sources_death_year_range"),
        ),
        sa.ForeignKeyConstraint(
            ["author_id"],
            ["authors.id"],
            name=op.f("fk_author_sources_author_id_authors"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("source", "source_author_id", name=op.f("pk_author_sources")),
    )
    op.create_table(
        "book_authors",
        sa.Column("book_id", sa.BigInteger(), nullable=False),
        sa.Column("author_id", sa.BigInteger(), nullable=False),
        sa.Column("position", sa.SmallInteger(), server_default=sa.text("0"), nullable=False),
        sa.ForeignKeyConstraint(
            ["author_id"],
            ["authors.id"],
            name=op.f("fk_book_authors_author_id_authors"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["book_id"],
            ["books.id"],
            name=op.f("fk_book_authors_book_id_books"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("book_id", "author_id", name=op.f("pk_book_authors")),
    )
    op.create_table(
        "book_sources",
        sa.Column("book_id", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("source_updated", sa.DateTime(timezone=True), nullable=True),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("payload_hash", sa.Text(), nullable=False),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source IN ('gutendex', 'openlibrary', 'googlebooks')",
            name=op.f("ck_book_sources_source_known"),
        ),
        sa.ForeignKeyConstraint(
            ["book_id"],
            ["books.id"],
            name=op.f("fk_book_sources_book_id_books"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("source", "source_id", name=op.f("pk_book_sources")),
    )
    op.create_index("idx_book_sources_book_id", "book_sources", ["book_id"], unique=False)
    op.create_table(
        "book_subjects",
        sa.Column("book_id", sa.BigInteger(), nullable=False),
        sa.Column("subject_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["book_id"],
            ["books.id"],
            name=op.f("fk_book_subjects_book_id_books"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["subject_id"],
            ["subjects.id"],
            name=op.f("fk_book_subjects_subject_id_subjects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("book_id", "subject_id", name=op.f("pk_book_subjects")),
    )
    op.create_table(
        "rejected_records",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("source_id", sa.Text(), nullable=True),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("rejection_code", sa.Text(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column(
            "rejected_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "stage IN ('extract', 'transform', 'load')",
            name=op.f("ck_rejected_records_stage_known"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.id"],
            name=op.f("fk_rejected_records_run_id_ingestion_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rejected_records")),
    )
    op.create_table(
        "run_topic_partitions",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("expected_partitions", sa.Integer(), nullable=False),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "topic IN ('books.raw', 'books.clean')",
            name=op.f("ck_run_topic_partitions_topic_known"),
        ),
        sa.CheckConstraint(
            "expected_partitions > 0",
            name=op.f("ck_run_topic_partitions_expected_partitions_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.id"],
            name=op.f("fk_run_topic_partitions_run_id_ingestion_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "topic", name=op.f("pk_run_topic_partitions")),
    )
    op.create_table(
        "source_runs",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("records_extracted", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'success', 'skipped', 'failed')",
            name=op.f("ck_source_runs_status_known"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.id"],
            name=op.f("fk_source_runs_run_id_ingestion_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "source", name=op.f("pk_source_runs")),
    )
    op.create_table(
        "run_partition_markers",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("partition", sa.Integer(), nullable=False),
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "partition >= 0", name=op.f("ck_run_partition_markers_partition_non_negative")
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "topic"],
            ["run_topic_partitions.run_id", "run_topic_partitions.topic"],
            name=op.f("fk_run_partition_markers_run_id_run_topic_partitions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "run_id", "topic", "partition", name=op.f("pk_run_partition_markers")
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("run_partition_markers")
    op.drop_table("source_runs")
    op.drop_table("run_topic_partitions")
    op.drop_table("rejected_records")
    op.drop_table("book_subjects")
    op.drop_index("idx_book_sources_book_id", table_name="book_sources")
    op.drop_table("book_sources")
    op.drop_table("book_authors")
    op.drop_table("author_sources")
    op.drop_table("subjects")
    op.drop_table("ingestion_runs")
    op.drop_index(
        "idx_books_title_trgm",
        table_name="books",
        postgresql_using="gin",
        postgresql_ops={"title": "gin_trgm_ops"},
    )
    op.drop_index("idx_books_title_keyset", table_name="books")
    op.drop_index("idx_books_search", table_name="books", postgresql_using="gin")
    op.drop_table("books")
    op.drop_index(
        "idx_authors_name_trgm",
        table_name="authors",
        postgresql_using="gin",
        postgresql_ops={"name": "gin_trgm_ops"},
    )
    op.drop_table("authors")

    # pg_trgm is intentionally not dropped. It is database-wide, cheap to
    # leave in place, and dropping it would break anything else using it.
