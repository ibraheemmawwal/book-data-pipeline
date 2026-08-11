"""Catalogue schema as SQLAlchemy Core metadata.

This is the single source of truth for the shape of the database. The Alembic
migration mirrors it, and an integration test compares the two by reflection —
metadata and migrations drifting apart is a classic way for a schema change to
pass CI and fail in production.

Core rather than the ORM throughout: the load layer does multi-row upserts and
recomputes canonical fields from provenance, which is set-shaped work that the
identity map would only get in the way of.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Computed,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    Table,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID

# Explicit naming so Alembic can autogenerate stable, reversible constraint
# operations; the default unnamed constraints cannot be dropped by name.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

SOURCE_NAMES = ("gutendex", "openlibrary", "googlebooks")
RUN_STATUSES = ("running", "processing", "success", "partial_success", "failed")
SOURCE_RUN_STATUSES = ("running", "success", "skipped", "failed")
REJECTION_STAGES = ("extract", "transform", "load")
MARKER_TOPICS = ("books.raw", "books.clean")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    rendered = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({rendered})"


# The weights are the ranking contract: a title match must outrank a match in
# the description, or search returns whatever mentions the word most often.
SEARCH_VECTOR = (
    "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
    "setweight(to_tsvector('english', coalesce(subtitle, '')), 'B') || "
    "setweight(to_tsvector('english', coalesce(description, '')), 'C')"
)

books = Table(
    "books",
    metadata,
    Column("id", BigInteger, primary_key=True),
    # Canonical identity: an ISBN key or a deterministic fallback digest.
    Column("identity_key", Text, nullable=False, unique=True),
    Column("isbn13", Text, unique=True),
    Column("title", Text, nullable=False),
    Column("subtitle", Text),
    Column("published_year", SmallInteger),
    Column("publisher", Text),
    Column("page_count", Integer),
    Column("download_count", BigInteger),
    Column("language", Text),
    Column("description", Text),
    Column("cover_url", Text),
    # Fingerprint of the canonical fields, so an unchanged record does not
    # touch updated_at and re-running the pipeline is genuinely a no-op.
    Column("content_hash", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("search_vector", TSVECTOR, Computed(SEARCH_VECTOR, persisted=True)),
    CheckConstraint("isbn13 IS NULL OR isbn13 ~ '^[0-9]{13}$'", name="isbn13_format"),
    CheckConstraint(
        "published_year IS NULL OR published_year BETWEEN 1400 AND 2100",
        name="published_year_range",
    ),
    CheckConstraint("page_count IS NULL OR page_count > 0", name="page_count_positive"),
    CheckConstraint(
        "download_count IS NULL OR download_count >= 0", name="download_count_non_negative"
    ),
    CheckConstraint("language IS NULL OR language ~ '^[a-z]{3}$'", name="language_format"),
)

Index("idx_books_search", books.c.search_vector, postgresql_using="gin")
# The default catalogue keyset is title-based: Gutendex supplies no publication
# year for the bulk dataset, so a year-ordered index would sort almost every
# row into one bucket.
Index("idx_books_title_keyset", func.lower(books.c.title), books.c.id)
Index(
    "idx_books_title_trgm",
    books.c.title,
    postgresql_using="gin",
    postgresql_ops={"title": "gin_trgm_ops"},
)

book_sources = Table(
    "book_sources",
    metadata,
    Column("book_id", BigInteger, ForeignKey("books.id", ondelete="CASCADE"), nullable=False),
    Column("source", Text, primary_key=True),
    Column("source_id", Text, primary_key=True),
    Column("source_updated", DateTime(timezone=True)),
    # The full source record. Canonical fields are recomputed from these on
    # every ingest, so provenance is not a debugging luxury — it is the input.
    Column("raw_payload", JSONB, nullable=False),
    Column("payload_hash", Text, nullable=False),
    Column("first_seen_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("last_seen_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint(_in_list("source", SOURCE_NAMES), name="source_known"),
)

Index("idx_book_sources_book_id", book_sources.c.book_id)

authors = Table(
    "authors",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("name", Text, nullable=False),
    Column("normalized_name", Text, nullable=False, unique=True),
    # Signed: Gutendex dates Homer to -750, so a non-negative bound would
    # reject real records from the primary source.
    Column("birth_year", SmallInteger),
    Column("death_year", SmallInteger),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint(
        "birth_year IS NULL OR birth_year BETWEEN -3000 AND 2100", name="birth_year_range"
    ),
    CheckConstraint(
        "death_year IS NULL OR death_year BETWEEN -3000 AND 2100", name="death_year_range"
    ),
    CheckConstraint(
        "birth_year IS NULL OR death_year IS NULL OR death_year >= birth_year",
        name="lifespan_order",
    ),
)

Index(
    "idx_authors_name_trgm",
    authors.c.name,
    postgresql_using="gin",
    postgresql_ops={"name": "gin_trgm_ops"},
)

author_sources = Table(
    "author_sources",
    metadata,
    Column("author_id", BigInteger, ForeignKey("authors.id", ondelete="CASCADE"), nullable=False),
    Column("source", Text, primary_key=True),
    Column("source_author_id", Text, primary_key=True),
    # Retained per source so a lifespan conflict between providers stays
    # auditable instead of being silently overwritten.
    Column("source_birth_year", SmallInteger),
    Column("source_death_year", SmallInteger),
    CheckConstraint(_in_list("source", SOURCE_NAMES), name="source_known"),
    CheckConstraint(
        "source_birth_year IS NULL OR source_birth_year BETWEEN -3000 AND 2100",
        name="birth_year_range",
    ),
    CheckConstraint(
        "source_death_year IS NULL OR source_death_year BETWEEN -3000 AND 2100",
        name="death_year_range",
    ),
    CheckConstraint(
        "source_birth_year IS NULL OR source_death_year IS NULL "
        "OR source_death_year >= source_birth_year",
        name="lifespan_order",
    ),
)

book_authors = Table(
    "book_authors",
    metadata,
    Column("book_id", BigInteger, ForeignKey("books.id", ondelete="CASCADE"), primary_key=True),
    Column("author_id", BigInteger, ForeignKey("authors.id", ondelete="CASCADE"), primary_key=True),
    Column("position", SmallInteger, nullable=False, server_default=text("0")),
)

subjects = Table(
    "subjects",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("name", Text, nullable=False),
    Column("normalized_name", Text, nullable=False, unique=True),
)

book_subjects = Table(
    "book_subjects",
    metadata,
    Column("book_id", BigInteger, ForeignKey("books.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "subject_id", BigInteger, ForeignKey("subjects.id", ondelete="CASCADE"), primary_key=True
    ),
)

ingestion_runs = Table(
    "ingestion_runs",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    # Airflow's run id, or cli:<uuid4> for a CLI run.
    Column("dag_run_id", Text, nullable=False, unique=True),
    Column("started_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("extraction_ended_at", DateTime(timezone=True)),
    Column("processing_ended_at", DateTime(timezone=True)),
    Column("status", Text, nullable=False, server_default=text("'running'")),
    Column("records_extracted", Integer, nullable=False, server_default=text("0")),
    Column("records_loaded", Integer, nullable=False, server_default=text("0")),
    Column("records_rejected", Integer, nullable=False, server_default=text("0")),
    CheckConstraint(_in_list("status", RUN_STATUSES), name="status_known"),
)

source_runs = Table(
    "source_runs",
    metadata,
    Column(
        "run_id",
        UUID(as_uuid=True),
        ForeignKey("ingestion_runs.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("source", Text, primary_key=True),
    Column("status", Text, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("finished_at", DateTime(timezone=True)),
    Column("records_extracted", Integer, nullable=False, server_default=text("0")),
    # Why a source was skipped or how it failed, so a gap in the catalogue is
    # explained in the run record rather than inferred.
    Column("error", Text),
    CheckConstraint(_in_list("status", SOURCE_RUN_STATUSES), name="status_known"),
)

# Broker topology frozen at barrier time. An event must never be an authority
# for topology: a marker carrying its own expectation would let a mis-produced
# event redefine what completion means.
run_topic_partitions = Table(
    "run_topic_partitions",
    metadata,
    Column(
        "run_id",
        UUID(as_uuid=True),
        ForeignKey("ingestion_runs.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("topic", Text, primary_key=True),
    Column("expected_partitions", Integer, nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint(_in_list("topic", MARKER_TOPICS), name="topic_known"),
    CheckConstraint("expected_partitions > 0", name="expected_partitions_positive"),
)

# Durable marker observation. Tracking these in process memory would leave a
# run in 'processing' forever after a consumer restart, with every book
# correctly loaded — which is what makes that failure so easy to miss.
run_partition_markers = Table(
    "run_partition_markers",
    metadata,
    Column("run_id", UUID(as_uuid=True), primary_key=True),
    Column("topic", Text, primary_key=True),
    Column("partition", Integer, primary_key=True),
    Column("observed_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("partition >= 0", name="partition_non_negative"),
    ForeignKeyConstraint(
        ["run_id", "topic"],
        ["run_topic_partitions.run_id", "run_topic_partitions.topic"],
        ondelete="CASCADE",
    ),
)

# Bad records are kept, not dropped. A pipeline that silently discards rows is
# one nobody can trust.
rejected_records = Table(
    "rejected_records",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column(
        "run_id",
        UUID(as_uuid=True),
        ForeignKey("ingestion_runs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("source", Text),
    Column("source_id", Text),
    Column("stage", Text, nullable=False),
    Column("raw_payload", JSONB, nullable=False),
    Column("rejection_code", Text, nullable=False),
    Column("detail", Text),
    Column("rejected_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint(_in_list("stage", REJECTION_STAGES), name="stage_known"),
)

__all__ = [
    "author_sources",
    "authors",
    "book_authors",
    "book_sources",
    "book_subjects",
    "books",
    "ingestion_runs",
    "metadata",
    "rejected_records",
    "run_partition_markers",
    "run_topic_partitions",
    "source_runs",
    "subjects",
]
