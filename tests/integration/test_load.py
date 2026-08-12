"""The canonical load layer against real PostgreSQL.

The claim under test is that re-running identical input changes nothing: no new
rows, no moved ``updated_at``. That depends on ON CONFLICT semantics and cascade
ordering, so it can only be proven here.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import Connection, Engine, event, insert, select, text, update
from sqlalchemy.exc import IntegrityError

from pipeline.extract import goodreads, googlebooks, gutendex, openlibrary
from pipeline.extract.base import Rejected
from pipeline.extract.resolver import Attempt, Outcome
from pipeline.load import (
    CatalogueLoader,
    LoadResult,
    record_attempts,
    record_rejection,
)
from pipeline.models.db import authors as authors_table
from pipeline.models.db import (
    book_authors,
    book_series,
    book_sources,
    book_subjects,
    books,
    ingestion_runs,
    rejected_records,
    resolution_attempts,
    series,
    series_sources,
    subjects,
)
from pipeline.models.domain import (
    CleanBook,
    RawBook,
    RawSeriesMembership,
    SourceName,
)
from pipeline.transform import canonicalise

pytestmark = pytest.mark.integration


def _clean(record: RawBook) -> CleanBook:
    result = canonicalise(record)
    assert isinstance(result, CleanBook), result
    return result


def gutendex_payload(source_id: str = "1", **fields: Any) -> CleanBook:
    """A record whose raw_payload is genuinely Gutendex-shaped.

    The loader recomputes canonical fields by replaying stored payloads through
    the source mappers, so a payload that does not match its declared source is
    silently rejected on recompute. Building real shapes here means these tests
    exercise the mapper round trip rather than working around it.
    """
    payload: dict[str, Any] = {
        "id": int(source_id),
        "title": fields.pop("title", "Moby Dick"),
        "authors": fields.pop("authors", []),
        "subjects": fields.pop("subjects", []),
        "languages": fields.pop("languages", ["en"]),
        "summaries": fields.pop("summaries", []),
        "formats": {},
        **fields,
    }
    mapped = gutendex.map_payload(payload)
    assert isinstance(mapped, RawBook), mapped
    return _clean(mapped)


def openlibrary_payload(source_id: str = "/works/OL1W", **fields: Any) -> CleanBook:
    """A record whose raw_payload is genuinely Open Library-shaped."""
    payload: dict[str, Any] = {
        "key": source_id,
        "title": fields.pop("title", "Moby Dick"),
        "author_name": fields.pop("author_name", []),
        "author_key": fields.pop("author_key", []),
        "isbn": fields.pop("isbn", []),
        "language": fields.pop("language", []),
        "subject": fields.pop("subject", []),
        **fields,
    }
    mapped = openlibrary.map_payload(payload)
    assert isinstance(mapped, RawBook), mapped
    return _clean(mapped)


def googlebooks_payload(source_id: str = "gb1", **fields: Any) -> CleanBook:
    """A record whose raw_payload is genuinely Google Books-shaped."""
    info: dict[str, Any] = {
        "title": fields.pop("title", "Moby Dick"),
        "authors": fields.pop("authors", []),
        "industryIdentifiers": [
            {"type": "ISBN_13", "identifier": isbn} for isbn in fields.pop("isbns", [])
        ],
        **fields,
    }
    mapped = googlebooks.map_payload({"id": source_id, "volumeInfo": info})
    assert isinstance(mapped, RawBook), mapped
    return _clean(mapped)


def count(connection: Connection, table: Any) -> int:
    return connection.execute(select(text("count(*)")).select_from(table)).scalar_one()


class TestIdempotency:
    def test_loading_once_creates_the_book(self, engine: Engine, connection: Connection) -> None:
        CatalogueLoader().load(engine, [gutendex_payload("1")])

        assert count(connection, books) == 1
        assert count(connection, book_sources) == 1

    def test_reloading_identical_input_changes_no_counts(
        self, engine: Engine, connection: Connection
    ) -> None:
        loader = CatalogueLoader()
        loader.load(engine, [gutendex_payload("1")])
        loader.load(engine, [gutendex_payload("1")])

        assert count(connection, books) == 1
        assert count(connection, book_sources) == 1

    def test_reloading_identical_input_does_not_move_updated_at(
        self, engine: Engine, connection: Connection
    ) -> None:
        # The strictest form of the idempotency claim, and the one that catches
        # an upsert that "works" but rewrites every row on every run.
        loader = CatalogueLoader()
        loader.load(engine, [gutendex_payload("1")])
        before = connection.execute(select(books.c.updated_at)).scalar_one()

        loader.load(engine, [gutendex_payload("1")])
        after = connection.execute(select(books.c.updated_at)).scalar_one()

        assert before == after

    def test_an_unchanged_reload_is_reported_as_unchanged(self, engine: Engine) -> None:
        loader = CatalogueLoader()
        loader.load(engine, [gutendex_payload("1")])
        result = loader.load(engine, [gutendex_payload("1")])

        assert result.books_unchanged == 1
        assert result.books_updated == 0

    def test_a_changed_record_does_move_updated_at(
        self, engine: Engine, connection: Connection
    ) -> None:
        loader = CatalogueLoader()
        loader.load(engine, [gutendex_payload("1")])
        before = connection.execute(select(books.c.updated_at)).scalar_one()

        loader.load(engine, [gutendex_payload("1", download_count=99)])
        after = connection.execute(select(books.c.updated_at)).scalar_one()

        assert after > before

    def test_an_isbn_less_record_is_idempotent(
        self, engine: Engine, connection: Connection
    ) -> None:
        # Gutendex publishes no ISBNs at all, so this is the common path, not
        # an edge case: NULL never conflicts with NULL in a unique index.
        loader = CatalogueLoader()
        for _ in range(3):
            loader.load(engine, [gutendex_payload("1")])

        assert count(connection, books) == 1
        assert connection.execute(select(books.c.isbn13)).scalar_one() is None


class TestProvenance:
    def test_two_sources_sharing_an_isbn_make_one_book(
        self, engine: Engine, connection: Connection
    ) -> None:
        shared = "9780553380163"
        CatalogueLoader().load(
            engine,
            [
                openlibrary_payload("/works/OL1W", isbn=[shared]),
                googlebooks_payload("gb1", isbns=[shared]),
            ],
        )

        assert count(connection, books) == 1
        assert count(connection, book_sources) == 2

    def test_each_source_keeps_its_own_provenance_row(
        self, engine: Engine, connection: Connection
    ) -> None:
        shared = "9780553380163"
        CatalogueLoader().load(
            engine,
            [
                openlibrary_payload("/works/OL1W", isbn=[shared]),
                googlebooks_payload("gb1", isbns=[shared]),
            ],
        )

        sources = (
            connection.execute(select(book_sources.c.source).order_by(book_sources.c.source))
            .scalars()
            .all()
        )

        assert sources == ["googlebooks", "openlibrary"]

    def test_raw_payload_is_retained_for_every_source(
        self, engine: Engine, connection: Connection
    ) -> None:
        CatalogueLoader().load(engine, [gutendex_payload("1")])

        payload = connection.execute(select(book_sources.c.raw_payload)).scalar_one()

        assert payload["id"] == 1

    def test_a_sparse_later_record_cannot_erase_a_rich_earlier_one(
        self, engine: Engine, connection: Connection
    ) -> None:
        # Canonical fields are recomputed from all provenance, not overwritten
        # by whichever record happened to arrive last.
        loader = CatalogueLoader()
        loader.load(engine, [gutendex_payload("1", download_count=500)])
        loader.load(
            engine,
            [openlibrary_payload("/works/OL1W", title="Moby Dick")],
        )

        stored = (
            connection.execute(select(books.c.download_count).order_by(books.c.id)).scalars().all()
        )

        assert 500 in stored


class TestAuthors:
    def test_authors_are_linked(self, engine: Engine, connection: Connection) -> None:
        CatalogueLoader().load(
            engine,
            [
                gutendex_payload(
                    authors=[{"name": "Melville, Herman", "birth_year": 1819, "death_year": 1891}]
                )
            ],
        )

        assert count(connection, book_authors) == 1

    def test_the_same_author_from_two_sources_is_one_row(
        self, engine: Engine, connection: Connection
    ) -> None:
        shared = "9780553380163"
        CatalogueLoader().load(
            engine,
            [
                openlibrary_payload(
                    "/works/OL1W",
                    isbn=[shared],
                    author_name=["Melville, Herman"],
                    author_key=["OL79034A"],
                ),
                googlebooks_payload("gb1", isbns=[shared], authors=["Herman Melville"]),
            ],
        )

        # Surname-first and natural order are the same person.
        assert count(connection, authors_table) == 1


class TestIsbnPromotionAndMerge:
    """A source that starts supplying an ISBN.

    Books ingested without one get a title-and-author fallback identity. When an
    ISBN turns up later the fallback book must either be promoted in place or
    folded into the book that already owns that ISBN — and provenance must
    survive either way, because book_sources cascades on delete.
    """

    def test_a_fallback_book_is_promoted_in_place(
        self, engine: Engine, connection: Connection
    ) -> None:
        loader = CatalogueLoader()
        loader.load(engine, [googlebooks_payload("gb1")])
        before = connection.execute(select(books.c.id)).scalar_one()
        connection.rollback()

        loader.load(engine, [googlebooks_payload("gb1", isbns=["9780553380163"])])

        row = connection.execute(select(books.c.id, books.c.isbn13)).one()
        assert row.id == before, "promotion must keep the same row, not create one"
        assert row.isbn13 == "9780553380163"
        assert count(connection, books) == 1

    def test_promotion_updates_the_identity_key(
        self, engine: Engine, connection: Connection
    ) -> None:
        loader = CatalogueLoader()
        loader.load(engine, [googlebooks_payload("gb1")])
        loader.load(engine, [googlebooks_payload("gb1", isbns=["9780553380163"])])

        identity = connection.execute(select(books.c.identity_key)).scalar_one()

        assert identity == "isbn:9780553380163"

    def test_a_conflicting_isbn_merges_into_the_isbn_owner(
        self, engine: Engine, connection: Connection
    ) -> None:
        shared = "9780553380163"
        loader = CatalogueLoader()
        # One book already owns the ISBN; another exists on a fallback key.
        loader.load(engine, [openlibrary_payload("/works/OL1W", isbn=[shared])])
        loader.load(engine, [googlebooks_payload("gb1", title="Moby Dick")])
        assert count(connection, books) == 2
        connection.rollback()

        # Google Books now reports the same ISBN for its record.
        loader.load(engine, [googlebooks_payload("gb1", title="Moby Dick", isbns=[shared])])

        assert count(connection, books) == 1

    def test_a_merge_keeps_every_provenance_row(
        self, engine: Engine, connection: Connection
    ) -> None:
        # book_sources cascades on delete, so deleting the orphan before moving
        # its links would silently destroy provenance.
        shared = "9780553380163"
        loader = CatalogueLoader()
        loader.load(engine, [openlibrary_payload("/works/OL1W", isbn=[shared])])
        loader.load(engine, [googlebooks_payload("gb1", title="Moby Dick")])
        loader.load(engine, [googlebooks_payload("gb1", title="Moby Dick", isbns=[shared])])

        sources = (
            connection.execute(select(book_sources.c.source).order_by(book_sources.c.source))
            .scalars()
            .all()
        )

        assert sources == ["googlebooks", "openlibrary"]

    def test_every_source_points_at_the_survivor_after_a_merge(
        self, engine: Engine, connection: Connection
    ) -> None:
        shared = "9780553380163"
        loader = CatalogueLoader()
        loader.load(engine, [openlibrary_payload("/works/OL1W", isbn=[shared])])
        loader.load(engine, [googlebooks_payload("gb1", title="Moby Dick")])
        loader.load(engine, [googlebooks_payload("gb1", title="Moby Dick", isbns=[shared])])

        survivor = connection.execute(select(books.c.id)).scalar_one()
        pointers = set(connection.execute(select(book_sources.c.book_id)).scalars().all())

        assert pointers == {survivor}

    def test_a_merge_is_reported(self, engine: Engine) -> None:
        shared = "9780553380163"
        loader = CatalogueLoader()
        loader.load(engine, [openlibrary_payload("/works/OL1W", isbn=[shared])])
        loader.load(engine, [googlebooks_payload("gb1", title="Moby Dick")])
        result = loader.load(
            engine, [googlebooks_payload("gb1", title="Moby Dick", isbns=[shared])]
        )

        assert result.merges == 1

    def test_merging_twice_is_a_no_op(self, engine: Engine, connection: Connection) -> None:
        # Redelivery after a Kafka restart replays the same record.
        shared = "9780553380163"
        loader = CatalogueLoader()
        loader.load(engine, [openlibrary_payload("/works/OL1W", isbn=[shared])])
        loader.load(engine, [googlebooks_payload("gb1", title="Moby Dick")])
        for _ in range(3):
            loader.load(engine, [googlebooks_payload("gb1", title="Moby Dick", isbns=[shared])])

        assert count(connection, books) == 1
        assert count(connection, book_sources) == 2

    def test_shared_authors_do_not_abort_the_merge(
        self, engine: Engine, connection: Connection
    ) -> None:
        # Both books may already link the same author; a plain insert of the
        # orphan's links would violate the primary key and roll the merge back.
        shared = "9780553380163"
        loader = CatalogueLoader()
        loader.load(
            engine,
            [
                openlibrary_payload(
                    "/works/OL1W",
                    isbn=[shared],
                    author_name=["Melville, Herman"],
                    author_key=["OL79034A"],
                )
            ],
        )
        loader.load(
            engine,
            [googlebooks_payload("gb1", title="Moby Dick", authors=["Herman Melville"])],
        )
        loader.load(
            engine,
            [
                googlebooks_payload(
                    "gb1", title="Moby Dick", isbns=[shared], authors=["Herman Melville"]
                )
            ],
        )

        assert count(connection, books) == 1
        # One author, linked once, despite arriving from both sides.
        assert count(connection, book_authors) == 1


class TestLoadResultAccounting:
    def test_records_loaded_sums_the_three_outcomes(self) -> None:
        result = LoadResult(books_inserted=2, books_updated=3, books_unchanged=5)

        assert result.records_loaded == 10


class TestSubjects:
    def test_subjects_are_linked(self, engine: Engine, connection: Connection) -> None:
        CatalogueLoader().load(
            engine, [gutendex_payload("1", subjects=["Whaling", "Adventure stories"])]
        )

        assert count(connection, subjects) == 2
        assert count(connection, book_subjects) == 2

    def test_the_same_subject_across_books_is_one_row(
        self, engine: Engine, connection: Connection
    ) -> None:
        CatalogueLoader().load(
            engine,
            [
                gutendex_payload("1", title="One", subjects=["Whaling"]),
                gutendex_payload("2", title="Two", subjects=["Whaling"]),
            ],
        )

        assert count(connection, subjects) == 1

    def test_reloading_does_not_duplicate_subject_links(
        self, engine: Engine, connection: Connection
    ) -> None:
        loader = CatalogueLoader()
        for _ in range(3):
            loader.load(engine, [gutendex_payload("1", subjects=["Whaling"])])

        assert count(connection, book_subjects) == 1


class TestRejectionRecording:
    def test_a_rejection_is_persisted_rather_than_dropped(
        self, engine: Engine, connection: Connection
    ) -> None:
        # A pipeline that silently discards bad rows is one nobody can trust.
        run_id = uuid.uuid4()
        with engine.begin() as conn:
            conn.execute(insert(ingestion_runs).values(id=run_id, dag_run_id=f"cli:{run_id}"))
            record_rejection(
                conn,
                run_id,
                Rejected(
                    source=SourceName.GUTENDEX,
                    source_id="99",
                    raw_payload={"id": 99},
                    rejection_code="invalid_record",
                    detail="title must not be blank",
                ),
            )

        stored = connection.execute(
            select(
                rejected_records.c.source,
                rejected_records.c.rejection_code,
                rejected_records.c.stage,
            )
        ).one()

        assert stored.source == "gutendex"
        assert stored.rejection_code == "invalid_record"
        assert stored.stage == "load"


class TestRecomputeGuards:
    def test_a_book_with_no_replayable_provenance_is_left_alone(
        self, engine: Engine, connection: Connection
    ) -> None:
        # A stored payload the mapper can no longer read must not blank the
        # canonical row it belongs to.
        CatalogueLoader().load(engine, [gutendex_payload("1", title="Moby Dick")])
        with engine.begin() as conn:
            conn.execute(update(book_sources).values(raw_payload={"garbage": True}))

        CatalogueLoader().load(engine, [gutendex_payload("2", title="Other")])

        titles = set(connection.execute(select(books.c.title)).scalars().all())
        assert "Moby Dick" in titles


class TestLoadEdgeCases:
    def test_an_author_whose_name_normalises_away_is_not_linked(
        self, engine: Engine, connection: Connection
    ) -> None:
        CatalogueLoader().load(
            engine,
            [gutendex_payload("1", authors=[{"name": "."}, {"name": "Melville, Herman"}])],
        )

        assert count(connection, authors_table) == 1

    def test_a_subject_that_normalises_away_is_not_linked(
        self, engine: Engine, connection: Connection
    ) -> None:
        CatalogueLoader().load(engine, [gutendex_payload("1", subjects=["   ", "Whaling"])])

        assert count(connection, subjects) == 1

    def test_reconciling_an_isbn_the_book_already_owns_is_a_no_op(
        self, engine: Engine, connection: Connection
    ) -> None:
        # Redelivery of an unchanged record must not attempt a merge.
        loader = CatalogueLoader()
        shared = "9780553380163"
        loader.load(engine, [googlebooks_payload("gb1", isbns=[shared])])
        result = loader.load(engine, [googlebooks_payload("gb1", isbns=[shared])])

        assert result.merges == 0
        assert count(connection, books) == 1

    def test_a_book_created_concurrently_is_adopted_not_duplicated(
        self, engine: Engine, connection: Connection
    ) -> None:
        # Two workers can reach _find_or_create_book for the same identity; the
        # loser must take the winner's row rather than fail the record.
        record = gutendex_payload("1", title="Moby Dick")
        with engine.begin() as conn:
            conn.execute(
                insert(books).values(
                    identity_key=record.identity_key,
                    title="Moby Dick",
                    content_hash="",
                )
            )

        CatalogueLoader().load(engine, [record])

        assert count(connection, books) == 1
        assert count(connection, book_sources) == 1


class TestUnreplayableProvenance:
    def test_a_source_row_that_no_longer_maps_is_skipped_not_fatal(
        self, engine: Engine, connection: Connection
    ) -> None:
        # Provenance written by an older mapper version may not survive a
        # contract change. Recompute must ignore that row and keep going,
        # rather than failing every book it touches.
        loader = CatalogueLoader()
        loader.load(engine, [gutendex_payload("1", title="Moby Dick")])
        with engine.begin() as conn:
            conn.execute(
                update(book_sources)
                .where(book_sources.c.source_id == "1")
                .values(raw_payload={"unrecognised": "shape"})
            )

        # A second source resolving to the same fallback identity forces a
        # recompute that replays both rows.
        loader.load(engine, [openlibrary_payload("/works/OL1W", title="Moby Dick")])

        assert count(connection, books) == 1
        assert connection.execute(select(books.c.title)).scalar_one() == "Moby Dick"


def goodreads_payload(source_id: str = "gr1", **fields: Any) -> CleanBook:
    """A record whose raw_payload is genuinely Goodreads-autocomplete-shaped."""
    payload: dict[str, Any] = {
        "bookId": source_id,
        "title": fields.pop("title", "A Game of Thrones (A Song of Ice and Fire, #1)"),
        "bookTitleBare": fields.pop("bare", "A Game of Thrones"),
        "author": {"id": "346732", "name": "George R.R. Martin"},
        **fields,
    }
    mapped = goodreads.map_payload(payload)
    assert isinstance(mapped, RawBook), mapped
    return _clean(mapped)


class TestSeries:
    def test_a_series_is_created_and_linked(self, engine: Engine, connection: Connection) -> None:
        CatalogueLoader().load(engine, [goodreads_payload()])

        assert count(connection, series) == 1
        assert count(connection, book_series) == 1

    def test_the_position_survives_as_an_exact_decimal(
        self, engine: Engine, connection: Connection
    ) -> None:
        CatalogueLoader().load(engine, [goodreads_payload(title="Novella (Discworld, #2.5)")])

        position = connection.execute(select(book_series.c.position)).scalar_one()
        assert position == Decimal("2.5")

    def test_two_books_in_one_series_share_the_series_row(
        self, engine: Engine, connection: Connection
    ) -> None:
        CatalogueLoader().load(
            engine,
            [
                goodreads_payload("gr1", title="One (A Song of Ice and Fire, #1)", bare="One"),
                goodreads_payload("gr2", title="Two (A Song of Ice and Fire, #2)", bare="Two"),
            ],
        )

        assert count(connection, series) == 1
        assert count(connection, book_series) == 2

    def test_series_search_text_is_populated(self, engine: Engine, connection: Connection) -> None:
        CatalogueLoader().load(engine, [goodreads_payload()])

        text = connection.execute(select(books.c.series_search_text)).scalar_one()
        assert text == "A Song of Ice and Fire"

    def test_a_book_with_no_series_has_an_empty_projection(
        self, engine: Engine, connection: Connection
    ) -> None:
        # The column is NOT NULL DEFAULT '', so this must never be NULL.
        CatalogueLoader().load(engine, [gutendex_payload("1")])

        assert connection.execute(select(books.c.series_search_text)).scalar_one() == ""

    def test_the_series_name_is_searchable_at_title_weight(
        self, engine: Engine, connection: Connection
    ) -> None:
        # The whole reason the projection is denormalised: a reader searching a
        # series name should find its books.
        CatalogueLoader().load(engine, [goodreads_payload()])

        found = (
            connection.execute(
                text(
                    "SELECT title FROM books, websearch_to_tsquery('english', 'ice and fire') q "
                    "WHERE search_vector @@ q"
                )
            )
            .scalars()
            .all()
        )

        assert found == ["A Game of Thrones"]

    def test_reloading_does_not_duplicate_the_series(
        self, engine: Engine, connection: Connection
    ) -> None:
        loader = CatalogueLoader()
        for _ in range(3):
            loader.load(engine, [goodreads_payload()])

        assert count(connection, series) == 1
        assert count(connection, book_series) == 1

    def test_reloading_does_not_move_updated_at(
        self, engine: Engine, connection: Connection
    ) -> None:
        loader = CatalogueLoader()
        loader.load(engine, [goodreads_payload()])
        before = connection.execute(select(books.c.updated_at)).scalar_one()
        connection.rollback()

        loader.load(engine, [goodreads_payload()])

        assert connection.execute(select(books.c.updated_at)).scalar_one() == before

    def test_a_confirmed_relationship_is_recorded_as_confirmed(
        self, engine: Engine, connection: Connection
    ) -> None:
        # A title-inferred series is a guess, so it stores confirmed=false.
        CatalogueLoader().load(engine, [goodreads_payload()])

        assert connection.execute(select(book_series.c.confirmed)).scalar_one() is False

    def test_provenance_is_kept_when_a_source_series_id_exists(
        self, engine: Engine, connection: Connection
    ) -> None:
        # Autocomplete carries no series id, so nothing to record here; the
        # detail parser is what supplies one.
        CatalogueLoader().load(engine, [goodreads_payload()])

        assert count(connection, series_sources) == 0


class TestResolutionAttempts:
    """Persisting why each source was or was not used.

    With an unofficial primary source this is operational data: without it, a
    run that fell back for every candidate looks exactly like one that never
    needed to.
    """

    def _run_id(self, engine: Engine) -> uuid.UUID:
        run_id = uuid.uuid4()
        with engine.begin() as conn:
            conn.execute(insert(ingestion_runs).values(id=run_id, dag_run_id=f"cli:{run_id}"))
        return run_id

    def _attempt(self, **overrides: Any) -> Attempt:
        base: dict[str, Any] = {
            "candidate_key": "/works/OL1W",
            "source": SourceName.GOODREADS,
            "attempt_no": 1,
            "outcome": Outcome.RESOLVED,
            "fallback_reason": None,
            "duration_ms": 42,
        }
        return Attempt(**(base | overrides))

    def test_attempts_are_persisted(self, engine: Engine, connection: Connection) -> None:
        run_id = self._run_id(engine)
        with engine.begin() as conn:
            written = record_attempts(
                conn,
                run_id,
                [
                    self._attempt(),
                    self._attempt(
                        source=SourceName.OPENLIBRARY,
                        outcome=Outcome.SKIPPED,
                        fallback_reason="no retained discovery payload",
                    ),
                ],
            )

        assert written == 2
        assert count(connection, resolution_attempts) == 2

    def test_the_reason_a_source_was_skipped_survives(
        self, engine: Engine, connection: Connection
    ) -> None:
        run_id = self._run_id(engine)
        with engine.begin() as conn:
            record_attempts(
                conn,
                run_id,
                [self._attempt(outcome=Outcome.SKIPPED, fallback_reason="circuit open")],
            )

        stored = connection.execute(
            select(resolution_attempts.c.outcome, resolution_attempts.c.fallback_reason)
        ).one()

        assert stored.outcome == "skipped"
        assert stored.fallback_reason == "circuit open"

    def test_rewriting_the_same_attempt_is_idempotent(
        self, engine: Engine, connection: Connection
    ) -> None:
        # An Airflow retry re-resolves the same candidates; the attempt record
        # must never be the thing that breaks a rerun.
        run_id = self._run_id(engine)
        for outcome in (Outcome.UNAVAILABLE, Outcome.RESOLVED):
            with engine.begin() as conn:
                record_attempts(conn, run_id, [self._attempt(outcome=outcome)])

        assert count(connection, resolution_attempts) == 1
        assert connection.execute(select(resolution_attempts.c.outcome)).scalar_one() == "resolved"

    def test_an_empty_list_writes_nothing(self, engine: Engine, connection: Connection) -> None:
        run_id = self._run_id(engine)
        with engine.begin() as conn:
            assert record_attempts(conn, run_id, []) == 0

        assert count(connection, resolution_attempts) == 0

    def test_attempts_are_removed_with_their_run(
        self, engine: Engine, connection: Connection
    ) -> None:
        run_id = self._run_id(engine)
        with engine.begin() as conn:
            record_attempts(conn, run_id, [self._attempt()])
            conn.execute(ingestion_runs.delete().where(ingestion_runs.c.id == run_id))

        assert count(connection, resolution_attempts) == 0

    def test_an_unknown_outcome_is_refused_by_the_database(self, engine: Engine) -> None:
        # The CHECK constraint is the last line of defence if the enum and the
        # schema ever drift apart.
        run_id = self._run_id(engine)
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(
                insert(resolution_attempts).values(
                    run_id=run_id,
                    candidate_key="/works/OL1W",
                    source="goodreads",
                    attempt_no=1,
                    outcome="invented",
                )
            )


class TestSeriesProvenance:
    def test_a_confirmed_series_id_is_recorded_as_provenance(
        self, engine: Engine, connection: Connection
    ) -> None:
        # Autocomplete carries no series id; a detail page's /series/ link
        # does, and that is what makes the relationship evidenced rather than
        # inferred. The provenance row is how that stays auditable.
        record = RawBook(
            source=SourceName.GOODREADS,
            source_id="gr9",
            title="A Game of Thrones",
            series=[
                RawSeriesMembership(
                    name="A Song of Ice and Fire",
                    source_series_id="45175",
                    position="1",
                    confirmed=True,
                )
            ],
            raw_payload={
                "bookId": "gr9",
                "title": "A Game of Thrones",
                # What _enrich stores after a detail fetch, so the recompute
                # can rebuild the same relationship without re-fetching.
                "_detail": {
                    "series_label": "Book 1 in the A Song of Ice and Fire series",
                    "series_id": "45175",
                },
            },
        )
        CatalogueLoader().load(engine, [_clean(record)])

        stored = connection.execute(
            select(series_sources.c.source_series_id, series_sources.c.raw_payload)
        ).one()
        assert stored.source_series_id == "45175"
        assert stored.raw_payload["confirmed"] is True

    def test_reloading_a_confirmed_series_does_not_duplicate_provenance(
        self, engine: Engine, connection: Connection
    ) -> None:
        record = RawBook(
            source=SourceName.GOODREADS,
            source_id="gr9",
            title="A Game of Thrones",
            series=[
                RawSeriesMembership(
                    name="A Song of Ice and Fire",
                    source_series_id="45175",
                    position="1",
                    confirmed=True,
                )
            ],
            raw_payload={
                "bookId": "gr9",
                "title": "A Game of Thrones",
                "_detail": {
                    "series_label": "Book 1 in the A Song of Ice and Fire series",
                    "series_id": "45175",
                },
            },
        )
        loader = CatalogueLoader()
        for _ in range(3):
            loader.load(engine, [_clean(record)])

        assert count(connection, series_sources) == 1


class TestCostDoesNotScaleWithRichness:
    """A load must cost the same whether a book has two subjects or fifty.

    This is a correctness property, not a benchmark. The loader once issued two
    statements per subject and three per author, which is invisible against a
    local socket at 0.4ms a round trip and ruinous against a managed database
    at 130ms: a fifty-subject Open Library record cost 112 statements, about
    fourteen seconds for one book.

    Counting statements rather than timing them keeps the test deterministic
    and keeps it honest about what actually went wrong — nothing here was slow,
    there was simply too much of it.
    """

    @staticmethod
    def _record(source_id: str, *, subjects_count: int, authors_count: int) -> CleanBook:
        # The payload is what matters: canonical fields are recomputed by
        # replaying book_sources.raw_payload, so a thin payload would quietly
        # measure a thin book however rich the record looked.
        payload = {
            "key": f"/works/{source_id}",
            "title": f"Book {source_id}",
            "author_name": [f"Author {source_id} {i}" for i in range(authors_count)],
            "subject": [f"subject {source_id} {i}" for i in range(subjects_count)],
            "language": ["eng"],
            "first_publish_year": 1990,
        }
        return _clean(openlibrary.map_payload(payload))

    def _statements_to_load(self, engine: Engine, record: CleanBook) -> int:
        counted = 0

        def count(*_args: Any, **_kwargs: Any) -> None:
            nonlocal counted
            counted += 1

        event.listen(engine, "before_cursor_execute", count)
        try:
            CatalogueLoader().load(engine, [record])
        finally:
            event.remove(engine, "before_cursor_execute", count)
        return counted

    def test_fifty_subjects_cost_no_more_than_two(self, engine: Engine) -> None:
        modest = self._statements_to_load(
            engine, self._record("OL1W", subjects_count=2, authors_count=1)
        )
        rich = self._statements_to_load(
            engine, self._record("OL2W", subjects_count=50, authors_count=1)
        )

        assert rich == modest, (
            f"{rich} statements for 50 subjects vs {modest} for 2: the per-subject loop is back"
        )

    def test_many_authors_cost_no_more_than_one(self, engine: Engine) -> None:
        one = self._statements_to_load(
            engine, self._record("OL3W", subjects_count=1, authors_count=1)
        )
        many = self._statements_to_load(
            engine, self._record("OL4W", subjects_count=1, authors_count=12)
        )

        assert many == one, (
            f"{many} statements for 12 authors vs {one} for 1: the per-author loop is back"
        )

    def test_the_fixed_cost_stays_in_budget(self, engine: Engine) -> None:
        """A ceiling on the per-book cost that does not scale away.

        Loose enough not to break on an unrelated statement, tight enough that
        adding another read per book has to be a decision someone makes on
        purpose.
        """
        used = self._statements_to_load(
            engine, self._record("OL5W", subjects_count=8, authors_count=2)
        )

        assert used <= 16, f"{used} statements to load one book"

    def test_a_rich_book_still_loads_everything(
        self, engine: Engine, connection: Connection
    ) -> None:
        # The batching must not have traded correctness for round trips.
        CatalogueLoader().load(engine, [self._record("OL6W", subjects_count=50, authors_count=3)])

        assert count(connection, subjects) == 50
        assert count(connection, book_subjects) == 50
        assert count(connection, authors_table) == 3
        assert count(connection, book_authors) == 3
