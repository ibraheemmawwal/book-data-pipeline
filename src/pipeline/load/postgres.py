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

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Connection, Engine, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import DBAPIError

from pipeline.extract import Rejected, map_payload
from pipeline.models.db import (
    author_sources,
    authors,
    book_authors,
    book_series,
    book_series_sources,
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
from pipeline.transform.identity import ISBN_PREFIX
from pipeline.transform.series import series_search_text

logger = structlog.get_logger(__name__)

# One transaction per batch. Small enough that a failure does not roll back an
# unbounded amount of work, large enough that per-statement overhead is noise.
DEFAULT_BATCH_SIZE = 1000

# PostgreSQL's own advice: a deadlock is a normal outcome of concurrent
# writers, not a bug to design away, and the remedy is to retry the
# transaction. Sorting each statement's rows makes it rarer — it cannot make it
# impossible, because a batch issues one insert per book and two transactions
# still interleave those in different orders.
#
# Retrying is only safe because loading is idempotent: the batch that half
# succeeded rolled back entirely, and replaying it writes the same rows.
DEADLOCK_RETRIES = 4
_DEADLOCK_SQLSTATES = frozenset({"40P01", "40001"})


def _is_deadlock(error: BaseException) -> bool:
    """Whether this is a lock conflict worth retrying rather than reporting."""
    sqlstate = getattr(getattr(error, "orig", None), "sqlstate", None)
    return sqlstate in _DEADLOCK_SQLSTATES


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
    "goodreads_average_rating",
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
            self._load_batch(engine, batch, total, run_id)
        return total

    def _load_batch(
        self,
        engine: Engine,
        batch: list[CleanBook],
        total: LoadResult,
        run_id: UUID | None,
    ) -> None:
        """One batch, one transaction, retried if it loses a deadlock.

        Counts accumulate into a throwaway result and are merged only once the
        transaction commits. Adding them as records are processed would count
        the half of a rolled-back batch that had already run, and a retry would
        then report more books than exist.
        """
        for attempt in range(DEADLOCK_RETRIES + 1):
            attempted = LoadResult()
            try:
                with engine.begin() as connection:
                    for record in batch:
                        self._load_one(connection, record, attempted, run_id)
            except DBAPIError as error:
                if not _is_deadlock(error) or attempt == DEADLOCK_RETRIES:
                    raise
                logger.warning(
                    "load.deadlock_retry",
                    attempt=attempt + 1,
                    records=len(batch),
                    detail="lost a lock race with another writer; replaying the batch",
                )
                # Brief, growing, and jittered by the batch's own size: two
                # writers that back off by the same amount simply collide again.
                time.sleep(0.05 * (attempt + 1) + (len(batch) % 7) * 0.01)
                continue

            total.books_inserted += attempted.books_inserted
            total.books_updated += attempted.books_updated
            total.books_unchanged += attempted.books_unchanged
            total.sources_linked += attempted.sources_linked
            total.merges += attempted.merges
            total.rejected += attempted.rejected
            total.rejections.extend(attempted.rejections)
            return

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

        # Series memberships are deliberately not moved. book_series.book_id
        # cascades on delete, so the orphan's links go with it — and they are
        # rebuilt a moment later, in this same transaction, because
        # book_sources moved first and _recompute derives series from the
        # payloads attached to the surviving book. Moving them by hand would
        # duplicate the recompute and give the merge a second, divergent
        # opinion about position and confirmation.
        #
        # This looks like data loss on a read of _merge alone, and has been
        # reported as such. TestAMergeKeepsTheSeries pins the outcome.
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

    def _replay(self, connection: Connection, book_id: int) -> tuple[Any, list[CleanBook]]:
        """The book's own row and every stored payload, in one round trip.

        These were two queries. Against a local socket that is 0.8ms and not
        worth a thought; against a managed database it is 120ms on every book
        the pipeline ever loads, and the loader is called once per record.

        Replaying provenance rather than trusting the record in hand is what
        stops a later sparse record erasing a richer earlier one, so the shape
        stays: the join just fetches the arbiter alongside the evidence.
        """
        rows = connection.execute(
            select(
                books.c.identity_key,
                books.c.content_hash,
                book_sources.c.source,
                book_sources.c.raw_payload,
            )
            .select_from(books.outerjoin(book_sources, book_sources.c.book_id == books.c.id))
            .where(books.c.id == book_id)
            .order_by(book_sources.c.source, book_sources.c.source_id)
        ).all()

        candidates: list[CleanBook] = []
        for row in rows:
            if row.source is None:
                # The outer join's placeholder for a book with no provenance.
                continue
            mapped = map_payload(SourceName(row.source), row.raw_payload)
            if isinstance(mapped, Rejected):
                continue
            cleaned = canonicalise(mapped)
            if isinstance(cleaned, CleanBook):
                candidates.append(cleaned)
        return (rows[0] if rows else None), candidates

    def _recompute(
        self,
        connection: Connection,
        book_id: int,
        result: LoadResult,
        run_id: UUID | None,
    ) -> None:
        """Recompute canonical fields and write only if they actually changed."""
        # Candidates can disagree on identity once a merge has pulled a
        # fallback book into an ISBN one. The book's own row is authoritative,
        # and is read alongside them rather than in a second round trip.
        current, candidates = self._replay(connection, book_id)
        if current is None or not candidates:
            return
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

        series_ids: dict[str, int] = {}
        for key, (membership, _source) in best.items():
            series_ids[key] = self._upsert_series(connection, membership)
            self._link_series(connection, book_id, series_ids[key], membership)

        # Provenance is every source's own account, not the one that won.
        #
        # The loop above collapses the candidates to a single merged opinion
        # per series, which is what book_series is for. That is the answer, and
        # it cannot say who gave it — so a reader asking whether a position was
        # stated or inferred, or whether two sources disagreed, has nothing to
        # read. book_series_sources is where the unmerged accounts live, and
        # until now nothing wrote to it.
        #
        # Only a membership carrying the source's own series id is recorded: the
        # row keys on it, and it is also the difference between a source that
        # named a series and one whose series was read out of a title. An
        # inferred membership still reaches book_series, still marked
        # unconfirmed, and simply has no provenance to show.
        for candidate in candidates:
            for membership in candidate.series:
                series_id = series_ids.get(membership.identity_key)
                if series_id is None or not membership.source_series_id:
                    continue
                self._record_series_source(connection, series_id, membership, candidate.source)
                self._link_series_source(connection, book_id, series_id, membership, candidate)

        self._refresh_series_search_text(connection, book_id)

    def _upsert_series(self, connection: Connection, membership: CleanSeriesMembership) -> int:
        """The canonical series row, independent of who reported it."""
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
        return int(row.id)

    @staticmethod
    def _membership_payload(membership: CleanSeriesMembership) -> dict[str, Any]:
        """What this source said, as it said it.

        ``position`` is stringified because it is a Decimal: JSON has one
        numeric type and it is binary, so 2.5 would survive and 0.10 would not.
        """
        return {
            "name": membership.name,
            "position": str(membership.position) if membership.position is not None else None,
            "confirmed": membership.confirmed,
        }

    def _record_series_source(
        self,
        connection: Connection,
        series_id: int,
        membership: CleanSeriesMembership,
        source: SourceName,
    ) -> None:
        """The source's own series row, which the provenance row references.

        Written for every source that named the series, not only the one whose
        account won, because book_series_sources keys on it and a foreign key
        to a row that was never written is a load failure.
        """
        payload = self._membership_payload(membership)
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

    def _link_series_source(
        self,
        connection: Connection,
        book_id: int,
        series_id: int,
        membership: CleanSeriesMembership,
        candidate: CleanBook,
    ) -> None:
        """One source's account of one book belonging to one series.

        Never merged with another source's: the point of this row is that it
        still says what it said after the canonical record has picked a winner.
        """
        payload = self._membership_payload(membership)
        statement = insert(book_series_sources).values(
            book_id=book_id,
            series_id=series_id,
            source=candidate.source.value,
            source_book_id=candidate.source_id,
            source_series_id=membership.source_series_id,
            position=membership.position,
            confirmed=membership.confirmed,
            raw_payload=payload,
        )
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=["source", "source_book_id", "source_series_id"],
                set_={
                    "book_id": statement.excluded.book_id,
                    "series_id": statement.excluded.series_id,
                    "position": statement.excluded.position,
                    "confirmed": statement.excluded.confirmed,
                    "raw_payload": statement.excluded.raw_payload,
                },
            )
        )

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

        The names and the stored projection are read together. They were two
        queries, which is one wasted round trip on every book in the catalogue
        — including the overwhelming majority that belong to no series at all.
        """
        current_names = (
            select(func.array_agg(series.c.name))
            .select_from(book_series.join(series, book_series.c.series_id == series.c.id))
            .where(book_series.c.book_id == book_id)
            .scalar_subquery()
        )
        row = connection.execute(
            select(books.c.series_search_text, current_names.label("names")).where(
                books.c.id == book_id
            )
        ).one()

        # array_agg over no rows is NULL, not an empty array.
        projection = series_search_text(list(row.names or []))
        if projection != row.series_search_text:
            connection.execute(
                update(books).where(books.c.id == book_id).values(series_search_text=projection)
            )

    def _sync_authors(
        self, connection: Connection, book_id: int, candidates: list[CleanBook]
    ) -> None:
        """Attach every author any source reported, in three statements.

        Same reasoning as ``_sync_subjects``: the per-author upsert was correct
        and its cost scaled with a number this loader does not control.
        """
        seen: dict[str, tuple[RawAuthor, SourceName, int]] = {}
        for candidate in candidates:
            for position, author in enumerate(candidate.authors):
                normalised = normalise_author(author.name)
                if normalised is None:
                    continue
                seen.setdefault(normalised, (author, candidate.source, position))

        if not seen:
            return

        statement = insert(authors).values(
            [
                {
                    "name": author.name,
                    "normalized_name": key,
                    "birth_year": author.birth_year,
                    "death_year": author.death_year,
                }
                for key, (author, _, _) in sorted(seen.items())
            ]
        )
        rows = connection.execute(
            statement.on_conflict_do_update(
                index_elements=["normalized_name"],
                # COALESCE so a source that omits a lifespan cannot blank one
                # another source supplied.
                set_={
                    "birth_year": func.coalesce(
                        authors.c.birth_year, statement.excluded.birth_year
                    ),
                    "death_year": func.coalesce(
                        authors.c.death_year, statement.excluded.death_year
                    ),
                },
            ).returning(authors.c.id, authors.c.normalized_name)
        ).all()
        # DO UPDATE touches every conflicting row, so every key comes back,
        # including the authors that already existed.
        author_ids = {str(row.normalized_name): int(row.id) for row in rows}

        connection.execute(
            insert(book_authors)
            .values(
                sorted(
                    (
                        {"book_id": book_id, "author_id": author_ids[key], "position": position}
                        for key, (_, _, position) in seen.items()
                    ),
                    key=lambda row: row["author_id"],
                )
            )
            .on_conflict_do_nothing(index_elements=["book_id", "author_id"])
        )

        source_rows = [
            {
                "author_id": author_ids[key],
                "source": source.value,
                "source_author_id": author.source_author_id,
                "source_birth_year": author.birth_year,
                "source_death_year": author.death_year,
            }
            for key, (author, source, _) in sorted(seen.items())
            if author.source_author_id
        ]
        if source_rows:
            # DO NOTHING tolerates a duplicate within the same statement, which
            # DO UPDATE would not: two candidates can name the same source id.
            connection.execute(
                insert(author_sources)
                .values(source_rows)
                .on_conflict_do_nothing(index_elements=["source", "source_author_id"])
            )

    def _sync_subjects(self, connection: Connection, book_id: int, names: Sequence[str]) -> None:
        """Attach every subject in two statements, whatever the count.

        This was a loop of two statements per subject. A rich Open Library
        record carries fifty of them, which is a hundred round trips for one
        book: 0.04s against a local socket and thirteen seconds against a
        managed database an ocean away. The work was never the problem, the
        number of times we asked was.
        """
        # Deduplicated on the normalised key before it becomes one statement.
        # A multi-row ON CONFLICT DO UPDATE that names the same key twice fails
        # outright ("cannot affect row a second time"); the loop this replaces
        # never noticed because it ran them one at a time.
        wanted: dict[str, str] = {}
        for name in names:
            normalised = normalise_subject(name)
            if normalised is not None:
                wanted.setdefault(normalised, name)

        if not wanted:
            return

        # Sorted by the conflict key, which is the whole point rather than
        # tidiness: a multi-row upsert takes its row locks in list order, and
        # two consumers writing overlapping subjects in different orders each
        # hold what the other is waiting for. That is a deadlock, and it took
        # three consumers and a live backlog to find it — one writer never
        # contends with itself, and the per-row loop this replaced could not
        # deadlock because each statement held exactly one lock.
        statement = insert(subjects).values(
            [{"name": name, "normalized_name": key} for key, name in sorted(wanted.items())]
        )
        rows = connection.execute(
            statement.on_conflict_do_update(
                index_elements=["normalized_name"],
                # A no-op update rather than DO NOTHING: RETURNING omits the
                # rows DO NOTHING skipped, and the ids of subjects that already
                # existed are precisely the ones needed here.
                set_={"normalized_name": statement.excluded.normalized_name},
            ).returning(subjects.c.id)
        ).all()

        connection.execute(
            insert(book_subjects)
            .values(
                [
                    {"book_id": book_id, "subject_id": subject_id}
                    for subject_id in sorted(int(row.id) for row in rows)
                ]
            )
            .on_conflict_do_nothing(index_elements=["book_id", "subject_id"])
        )


def _agree_on_identity(candidates: list[CleanBook], identity_key: str) -> list[CleanBook]:
    """Force candidates onto the book's identity before merging.

    After a merge the attached sources genuinely disagree — that disagreement is
    what caused the merge — and ``merge_candidates`` refuses to span identities
    by design. The stored row is the arbiter.
    """
    # CleanBook requires identity_key and isbn13 to agree, so they move
    # together; the source's own view survives in its raw_payload.
    target_isbn = (
        identity_key.removeprefix(ISBN_PREFIX) if identity_key.startswith(ISBN_PREFIX) else None
    )
    return [
        candidate
        if candidate.identity_key == identity_key
        else candidate.model_copy(update={"identity_key": identity_key, "isbn13": target_isbn})
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
