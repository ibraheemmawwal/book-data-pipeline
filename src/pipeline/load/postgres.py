"""Idempotent load into the canonical catalogue.

The contract is that re-running identical input changes nothing: no new rows,
no moved ``updated_at``. Everything here exists to make that true even when the
same book arrives from three providers, in any order, across runs that crashed
halfway.

Two identities do the work. ``(source, source_id)`` is the *ingestion* identity
— always present, and the conflict target for provenance rows. ``identity_key``
is the *canonical* identity — an ISBN where one is trustworthy, otherwise a
digest of title, author and year.

Canonical fields are never taken from the record in hand. They are recomputed
from every ``book_sources`` row attached to the book, by replaying stored raw
payloads through the source mappers. That is what makes a late-arriving sparse
record unable to erase a rich one, and it is why provenance is an input rather
than a debugging luxury.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Connection, Engine, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert

from pipeline.extract import Rejected, map_payload
from pipeline.models.db import (
    author_sources,
    authors,
    book_authors,
    book_series,
    book_sources,
    book_subjects,
    books,
    rejected_records,
    resolution_attempts,
    series,
    series_sources,
    subjects,
)
from pipeline.models.domain import CleanBook, CleanSeriesMembership, RawAuthor, SourceName
from pipeline.transform import (
    canonicalise,
    content_hash,
    merge_candidates,
    normalise_author,
    normalise_subject,
    payload_hash,
)
from pipeline.transform.series import series_search_text

logger = structlog.get_logger(__name__)

# One transaction per batch. Small enough that a failure does not roll back an
# unbounded amount of work, large enough that per-statement overhead is noise.
DEFAULT_BATCH_SIZE = 1000

# Mirrors the columns recomputed from provenance. Kept explicit so adding a
# canonical column is a deliberate decision rather than an accident of dir().
CANONICAL_FIELDS = (
    "title",
    "subtitle",
    "isbn13",
    "published_year",
    "publisher",
    "page_count",
    "download_count",
    "language",
    "description",
    "cover_url",
)


@dataclass
class LoadResult:
    """What one load pass did, for ``ingestion_runs``."""

    books_inserted: int = 0
    books_updated: int = 0
    books_unchanged: int = 0
    sources_linked: int = 0
    merges: int = 0
    rejected: int = 0
    rejections: list[Rejected] = field(default_factory=list)

    @property
    def records_loaded(self) -> int:
        return self.books_inserted + self.books_updated + self.books_unchanged


def _batched(records: Iterable[CleanBook], size: int) -> Iterable[list[CleanBook]]:
    iterator = iter(records)
    while batch := list(islice(iterator, size)):
        yield batch


class CatalogueLoader:
    """Writes clean records into the canonical catalogue, idempotently."""

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self._batch_size = min(batch_size, DEFAULT_BATCH_SIZE)

    def load(
        self,
        engine: Engine,
        records: Iterable[CleanBook],
        *,
        run_id: UUID | None = None,
    ) -> LoadResult:
        """Load records in batches, one transaction per batch.

        Takes an ``Engine`` rather than a ``Connection`` because the batch
        transaction boundary is part of this class's contract. Accepting a
        connection would make correctness depend on whether the caller happened
        to have a transaction open, and a partially-committed batch is exactly
        what the merge logic must never leave behind.
        """
        total = LoadResult()
        for batch in _batched(records, self._batch_size):
            with engine.begin() as connection:
                for record in batch:
                    self._load_one(connection, record, total, run_id)
        return total

    # -- one record ------------------------------------------------------

    def _load_one(
        self,
        connection: Connection,
        record: CleanBook,
        result: LoadResult,
        run_id: UUID | None,
    ) -> None:
        existing_book_id = self._locked_book_id_for_source(connection, record)

        if existing_book_id is not None:
            # A source row already points somewhere. Honour that: letting a
            # changed title move a record to a different canonical book would
            # duplicate the book and orphan its provenance.
            book_id = existing_book_id
            if record.isbn13:
                book_id = self._reconcile_isbn(connection, book_id, record, result)
        else:
            book_id = self._find_or_create_book(connection, record, result)

        self._upsert_source(connection, book_id, record, result)
        self._recompute(connection, book_id, result, run_id)

    def _locked_book_id_for_source(self, connection: Connection, record: CleanBook) -> int | None:
        """Read and lock the provenance row for this ingestion identity."""
        row = connection.execute(
            select(book_sources.c.book_id)
            .where(
                book_sources.c.source == record.source.value,
                book_sources.c.source_id == record.source_id,
            )
            .with_for_update()
        ).first()
        return int(row.book_id) if row else None

    def _find_or_create_book(
        self, connection: Connection, record: CleanBook, result: LoadResult
    ) -> int:
        """Locate the canonical book for this identity, creating a stub if new.

        The stub carries a placeholder ``content_hash``; ``_recompute`` fills in
        the real canonical fields from provenance a moment later, once this
        record's source row exists.
        """
        row = connection.execute(
            select(books.c.id).where(books.c.identity_key == record.identity_key).with_for_update()
        ).first()
        if row:
            return int(row.id)

        inserted = connection.execute(
            insert(books)
            .values(
                identity_key=record.identity_key,
                isbn13=record.isbn13,
                title=record.title,
                content_hash="",
            )
            # Another transaction may have created it between the select and
            # here; take theirs rather than failing the record.
            .on_conflict_do_nothing(index_elements=["identity_key"])
            .returning(books.c.id)
        ).first()

        if inserted is None:
            existing = connection.execute(
                select(books.c.id).where(books.c.identity_key == record.identity_key)
            ).one()
            return int(existing.id)

        result.books_inserted += 1
        return int(inserted.id)

    # -- ISBN promotion and merge ----------------------------------------

    def _reconcile_isbn(
        self,
        connection: Connection,
        book_id: int,
        record: CleanBook,
        result: LoadResult,
    ) -> int:
        """Handle a source that has started supplying an ISBN.

        Either the current book is promoted to the ISBN identity, or the ISBN
        already belongs to another book and the two must be merged. Returns the
        id of the book that survives.
        """
        current = connection.execute(
            select(books.c.id, books.c.identity_key, books.c.isbn13).where(books.c.id == book_id)
        ).one()

        if current.isbn13 == record.isbn13:
            return book_id

        owner = connection.execute(
            select(books.c.id).where(books.c.isbn13 == record.isbn13)
        ).first()

        if owner is None:
            # Nobody else claims it: promote in place, keeping the same row so
            # existing provenance and relationships stay attached.
            connection.execute(
                update(books)
                .where(books.c.id == book_id)
                .values(isbn13=record.isbn13, identity_key=record.identity_key)
            )
            return book_id

        if int(owner.id) == book_id:
            return book_id

        return self._merge(connection, survivor_id=int(owner.id), orphan_id=book_id, result=result)

    def _merge(
        self,
        connection: Connection,
        *,
        survivor_id: int,
        orphan_id: int,
        result: LoadResult,
    ) -> int:
        """Fold the orphan book into the ISBN-identified survivor.

        The order is the contract. Source links move *before* anything is
        deleted, because ``book_sources.book_id`` cascades on delete and
        removing the orphan first would take the provenance with it.
        """
        # Ascending id order: two concurrent merges touching the same pair must
        # not take their locks in opposite orders and deadlock.
        first, second = sorted((survivor_id, orphan_id))
        connection.execute(
            select(books.c.id)
            .where(books.c.id.in_((first, second)))
            .order_by(books.c.id)
            .with_for_update()
        ).all()

        connection.execute(
            update(book_sources)
            .where(book_sources.c.book_id == orphan_id)
            .values(book_id=survivor_id)
        )

        for link, conflict_columns in (
            (book_authors, ["book_id", "author_id"]),
            (book_subjects, ["book_id", "subject_id"]),
        ):
            other = "author_id" if link is book_authors else "subject_id"
            # DO NOTHING because the two books may already share an author or
            # subject; a plain insert would abort the whole merge.
            connection.execute(
                insert(link)
                .from_select(
                    ["book_id", other],
                    select(
                        select(books.c.id).where(books.c.id == survivor_id).scalar_subquery(),
                        link.c[other],
                    ).where(link.c.book_id == orphan_id),
                )
                .on_conflict_do_nothing(index_elements=conflict_columns)
            )
            connection.execute(delete(link).where(link.c.book_id == orphan_id))

        connection.execute(delete(books).where(books.c.id == orphan_id))

        result.merges += 1
        logger.info("load.books_merged", survivor=survivor_id, orphan=orphan_id)
        return survivor_id

    # -- provenance -------------------------------------------------------

    def _upsert_source(
        self,
        connection: Connection,
        book_id: int,
        record: CleanBook,
        result: LoadResult,
    ) -> None:
        """Record or refresh this source's view of the book."""
        digest = payload_hash(record.raw_payload)
        statement = insert(book_sources).values(
            book_id=book_id,
            source=record.source.value,
            source_id=record.source_id,
            source_updated=record.source_updated,
            raw_payload=record.raw_payload,
            payload_hash=digest,
        )
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=["source", "source_id"],
                set_={
                    "book_id": statement.excluded.book_id,
                    "raw_payload": statement.excluded.raw_payload,
                    "payload_hash": statement.excluded.payload_hash,
                    "source_updated": statement.excluded.source_updated,
                    # Observation time always moves; it records that we saw the
                    # record again, not that the record changed.
                    "last_seen_at": statement.excluded.first_seen_at,
                },
            )
        )
        result.sources_linked += 1

    # -- canonical recomputation ------------------------------------------

    def _candidates(self, connection: Connection, book_id: int) -> list[CleanBook]:
        """Replay every stored payload for this book through its source mapper.

        Recomputing from provenance rather than from the record in hand is what
        stops a later sparse record from erasing a richer earlier one.
        """
        rows = connection.execute(
            select(book_sources.c.source, book_sources.c.raw_payload)
            .where(book_sources.c.book_id == book_id)
            .order_by(book_sources.c.source, book_sources.c.source_id)
        ).all()

        candidates: list[CleanBook] = []
        for row in rows:
            mapped = map_payload(SourceName(row.source), row.raw_payload)
            if isinstance(mapped, Rejected):
                continue
            cleaned = canonicalise(mapped)
            if isinstance(cleaned, CleanBook):
                candidates.append(cleaned)
        return candidates

    def _recompute(
        self,
        connection: Connection,
        book_id: int,
        result: LoadResult,
        run_id: UUID | None,
    ) -> None:
        """Recompute canonical fields and write only if they actually changed."""
        candidates = self._candidates(connection, book_id)
        if not candidates:
            return

        # Candidates can disagree on identity once a merge has pulled a
        # fallback book into an ISBN one. The book's own row is authoritative.
        current = connection.execute(
            select(books.c.identity_key, books.c.content_hash).where(books.c.id == book_id)
        ).one()
        merged = merge_candidates(_agree_on_identity(candidates, current.identity_key))

        values: dict[str, Any] = {name: getattr(merged, name) for name in CANONICAL_FIELDS}
        # download_count means different things at different providers, so only
        # the Gutendex value is ever promoted to the canonical column.
        values["download_count"] = next(
            (c.download_count for c in candidates if c.source is SourceName.GUTENDEX),
            None,
        )
        digest = content_hash(values)

        if digest == current.content_hash:
            result.books_unchanged += 1
        else:
            connection.execute(
                update(books)
                .where(books.c.id == book_id)
                .values(**values, content_hash=digest, updated_at=func.now())
            )
            # A stub created moments ago is an insert, not an update.
            if current.content_hash:
                result.books_updated += 1

        self._sync_authors(connection, book_id, candidates)
        self._sync_subjects(connection, book_id, merged.subjects)
        self._sync_series(connection, book_id, candidates)
        _ = run_id

    def _sync_series(
        self, connection: Connection, book_id: int, candidates: list[CleanBook]
    ) -> None:
        """Attach every series relationship any source reported.

        A confirmed relationship wins over an inferred one for the same series:
        a ``/series/`` link that agreed with the parsed name is evidence, and a
        name pulled out of a title is a guess.
        """
        best: dict[str, tuple[CleanSeriesMembership, SourceName]] = {}
        for candidate in candidates:
            for membership in candidate.series:
                existing = best.get(membership.identity_key)
                if existing is None or (membership.confirmed and not existing[0].confirmed):
                    best[membership.identity_key] = (membership, candidate.source)

        for membership, source in best.values():
            series_id = self._upsert_series(connection, membership, source)
            self._link_series(connection, book_id, series_id, membership)

        self._refresh_series_search_text(connection, book_id)

    def _upsert_series(
        self,
        connection: Connection,
        membership: CleanSeriesMembership,
        source: SourceName,
    ) -> int:
        row = connection.execute(
            insert(series)
            .values(
                identity_key=membership.identity_key,
                name=membership.name,
                normalized_name=membership.normalised_name,
            )
            .on_conflict_do_update(
                index_elements=["identity_key"],
                set_={"normalized_name": membership.normalised_name},
            )
            .returning(series.c.id)
        ).one()
        series_id = int(row.id)

        if membership.source_series_id:
            payload = {
                "name": membership.name,
                "position": str(membership.position) if membership.position else None,
                "confirmed": membership.confirmed,
            }
            statement = insert(series_sources).values(
                series_id=series_id,
                source=source.value,
                source_series_id=membership.source_series_id,
                raw_payload=payload,
                payload_hash=payload_hash(payload),
            )
            connection.execute(
                statement.on_conflict_do_update(
                    index_elements=["source", "source_series_id"],
                    set_={
                        "series_id": statement.excluded.series_id,
                        "raw_payload": statement.excluded.raw_payload,
                        "payload_hash": statement.excluded.payload_hash,
                    },
                )
            )
        return series_id

    def _link_series(
        self,
        connection: Connection,
        book_id: int,
        series_id: int,
        membership: CleanSeriesMembership,
    ) -> None:
        statement = insert(book_series).values(
            book_id=book_id,
            series_id=series_id,
            position=membership.position,
            confirmed=membership.confirmed,
        )
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=["book_id", "series_id"],
                set_={
                    # COALESCE so a source that omits a position cannot blank
                    # one another source supplied.
                    "position": func.coalesce(statement.excluded.position, book_series.c.position),
                    # Confirmation is monotonic: once something confirmed the
                    # relationship, a later guess cannot un-confirm it.
                    "confirmed": book_series.c.confirmed | statement.excluded.confirmed,
                },
            )
        )

    def _refresh_series_search_text(self, connection: Connection, book_id: int) -> None:
        """Recompute the denormalised series projection for one book.

        Written only when it changes, so a book whose series did not move keeps
        its ``updated_at`` and the idempotency guarantee holds.
        """
        names = (
            connection.execute(
                select(series.c.name)
                .select_from(book_series.join(series, book_series.c.series_id == series.c.id))
                .where(book_series.c.book_id == book_id)
            )
            .scalars()
            .all()
        )
        projection = series_search_text(list(names))
        current = connection.execute(
            select(books.c.series_search_text).where(books.c.id == book_id)
        ).scalar_one()
        if projection != current:
            connection.execute(
                update(books).where(books.c.id == book_id).values(series_search_text=projection)
            )

    def _sync_authors(
        self, connection: Connection, book_id: int, candidates: list[CleanBook]
    ) -> None:
        """Attach every author any source reported, keeping per-source values."""
        seen: dict[str, tuple[RawAuthor, SourceName, int]] = {}
        for candidate in candidates:
            for position, author in enumerate(candidate.authors):
                normalised = normalise_author(author.name)
                if normalised is None:
                    continue
                seen.setdefault(normalised, (author, candidate.source, position))

        for normalised, (author, source, position) in seen.items():
            author_id = self._upsert_author(connection, author, normalised, source)
            connection.execute(
                insert(book_authors)
                .values(book_id=book_id, author_id=author_id, position=position)
                .on_conflict_do_nothing(index_elements=["book_id", "author_id"])
            )

    def _upsert_author(
        self,
        connection: Connection,
        author: RawAuthor,
        normalised: str,
        source: SourceName,
    ) -> int:
        row = connection.execute(
            insert(authors)
            .values(
                name=author.name,
                normalized_name=normalised,
                birth_year=author.birth_year,
                death_year=author.death_year,
            )
            .on_conflict_do_update(
                index_elements=["normalized_name"],
                # COALESCE so a source that omits a lifespan cannot blank one
                # another source supplied.
                set_={
                    "birth_year": func.coalesce(authors.c.birth_year, author.birth_year),
                    "death_year": func.coalesce(authors.c.death_year, author.death_year),
                },
            )
            .returning(authors.c.id)
        ).one()
        author_id = int(row.id)

        if author.source_author_id:
            connection.execute(
                insert(author_sources)
                .values(
                    author_id=author_id,
                    source=source.value,
                    source_author_id=author.source_author_id,
                    source_birth_year=author.birth_year,
                    source_death_year=author.death_year,
                )
                .on_conflict_do_nothing(index_elements=["source", "source_author_id"])
            )
        return author_id

    def _sync_subjects(self, connection: Connection, book_id: int, names: Sequence[str]) -> None:
        for name in names:
            normalised = normalise_subject(name)
            if normalised is None:
                continue
            row = connection.execute(
                insert(subjects)
                .values(name=name, normalized_name=normalised)
                .on_conflict_do_update(
                    index_elements=["normalized_name"],
                    set_={"normalized_name": normalised},
                )
                .returning(subjects.c.id)
            ).one()
            connection.execute(
                insert(book_subjects)
                .values(book_id=book_id, subject_id=int(row.id))
                .on_conflict_do_nothing(index_elements=["book_id", "subject_id"])
            )


def _agree_on_identity(candidates: list[CleanBook], identity_key: str) -> list[CleanBook]:
    """Force candidates onto the book's identity before merging.

    After a merge the attached sources genuinely disagree — that disagreement is
    what caused the merge — and ``merge_candidates`` refuses to span identities
    by design. The stored row is the arbiter.
    """
    return [
        candidate
        if candidate.identity_key == identity_key
        else candidate.model_copy(update={"identity_key": identity_key})
        for candidate in candidates
    ]


def record_rejection(
    connection: Connection, run_id: UUID, rejection: Rejected, stage: str = "load"
) -> None:
    """Persist a rejected record instead of dropping it."""
    connection.execute(
        insert(rejected_records).values(
            run_id=run_id,
            source=rejection.source.value if rejection.source else None,
            source_id=rejection.source_id,
            stage=stage,
            raw_payload=rejection.raw_payload,
            rejection_code=rejection.rejection_code,
            detail=rejection.detail,
        )
    )


def record_attempts(connection: Connection, run_id: UUID, attempts: Iterable[Any]) -> int:
    """Persist one run's resolution attempts.

    Idempotent on ``(run_id, candidate_key, source, attempt_no)`` so an Airflow
    retry that re-resolves the same candidates updates its own rows rather than
    failing the task on a primary-key clash — the attempt record should never
    be the thing that breaks a rerun.
    """
    rows = [
        {
            "run_id": run_id,
            "candidate_key": attempt.candidate_key,
            "source": attempt.source.value,
            "attempt_no": attempt.attempt_no,
            "outcome": attempt.outcome.value,
            "fallback_reason": attempt.fallback_reason,
            "duration_ms": attempt.duration_ms,
        }
        for attempt in attempts
    ]
    if not rows:
        return 0

    statement = insert(resolution_attempts).values(rows)
    connection.execute(
        statement.on_conflict_do_update(
            index_elements=["run_id", "candidate_key", "source", "attempt_no"],
            set_={
                "outcome": statement.excluded.outcome,
                "fallback_reason": statement.excluded.fallback_reason,
                "duration_ms": statement.excluded.duration_ms,
            },
        )
    )
    return len(rows)
