"""Finding contested books, and re-resolving them, against real PostgreSQL.

Which books qualify is decided by a SQL aggregate over stored payloads, so it
cannot be proven anywhere else. The resolution loop is exercised with a stubbed
tie-breaker: what is under test is what the loop does with an answer, not the
scrape.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import Connection, Engine, select

from pipeline.config import Settings
from pipeline.contested import find_contested, resolve_contested
from pipeline.extract import googlebooks, openlibrary
from pipeline.extract.base import Rejected
from pipeline.extract.goodreads import GoodreadsNotAcceptedError, GoodreadsUnavailableError
from pipeline.load import CatalogueLoader
from pipeline.models.db import book_sources, books, ingestion_runs
from pipeline.models.domain import CleanBook, RawBook, SourceName
from pipeline.transform import canonicalise

pytestmark = pytest.mark.integration


def _always(answer: Any) -> Any:
    async def fake(*_a: Any, **_k: Any) -> Any:
        return answer

    return fake


def _clean(record: RawBook) -> CleanBook:
    result = canonicalise(record)
    assert isinstance(result, CleanBook), result
    return result


def openlibrary_view(isbn: str, *, title: str, year: str, pages: int | None = 200) -> CleanBook:
    payload: dict[str, Any] = {
        "key": f"/works/{isbn}",
        "title": title,
        "isbn": [isbn],
        "first_publish_year": int(year),
        "number_of_pages_median": pages,
        "language": ["eng"],
    }
    return _clean(openlibrary.map_payload(payload))


def googlebooks_view(isbn: str, *, title: str, year: str, pages: int | None = 200) -> CleanBook:
    # Google Books nests everything under volumeInfo, which is precisely the
    # shape a flat comparison would read as "no disagreement".
    payload: dict[str, Any] = {
        "id": f"gb-{isbn}",
        "volumeInfo": {
            "title": title,
            "publishedDate": year,
            "pageCount": pages,
            "industryIdentifiers": [{"type": "ISBN_13", "identifier": isbn}],
            "language": "en",
        },
    }
    return _clean(googlebooks.map_payload(payload))


def settings_for(url: str, *, accepted: bool = True) -> Settings:
    return Settings(  # type: ignore[call-arg]
        database_url=url,
        openlibrary_contact_email="t@example.com",
        goodreads_enabled=True,
        goodreads_unofficial_source_accepted=accepted,
    )


class TestFindingContestedBooks:
    def test_agreeing_sources_are_not_contested(self, engine: Engine) -> None:
        isbn = "9780000000019"
        CatalogueLoader().load(
            engine,
            [
                openlibrary_view(isbn, title="Dune", year="1965"),
                googlebooks_view(isbn, title="Dune", year="1965"),
            ],
        )

        assert find_contested(engine, minimum_conflicts=1, limit=10) == []

    def test_disagreeing_sources_are_contested(self, engine: Engine) -> None:
        isbn = "9780000000026"
        CatalogueLoader().load(
            engine,
            [
                openlibrary_view(isbn, title="Dune", year="1965", pages=412),
                googlebooks_view(isbn, title="Dune: Special", year="1990", pages=896),
            ],
        )

        found = find_contested(engine, minimum_conflicts=2, limit=10)

        assert len(found) == 1
        assert found[0]["isbn13"] == isbn
        assert found[0]["conflicts"] >= 2
        assert sorted(found[0]["sources"]) == ["googlebooks", "openlibrary"]

    def test_a_single_source_book_is_never_contested(self, engine: Engine) -> None:
        # One source cannot disagree with itself, and the HAVING clause is what
        # keeps a catalogue of mostly-single-source books out of the results.
        CatalogueLoader().load(
            engine, [openlibrary_view("9780000000033", title="Solo", year="1970")]
        )

        assert find_contested(engine, minimum_conflicts=1, limit=10) == []

    def test_the_worst_records_come_first(self, engine: Engine) -> None:
        """Ordering is what makes a bounded run worth doing.

        A limit that spent its budget on whichever books were inserted first
        would be a sample, not a triage.
        """
        mild, severe = "9780000000040", "9780000000057"
        CatalogueLoader().load(
            engine,
            [
                openlibrary_view(mild, title="Mild", year="1965", pages=300),
                googlebooks_view(mild, title="Mild", year="1966", pages=300),
                openlibrary_view(severe, title="Severe", year="1965", pages=100),
                googlebooks_view(severe, title="Severe Edition", year="1999", pages=900),
            ],
        )

        found = find_contested(engine, minimum_conflicts=1, limit=10)

        assert [item["isbn13"] for item in found] == [severe, mild]

    def test_the_limit_is_honoured(self, engine: Engine) -> None:
        # Valid check digits: an ISBN that fails validation is never promoted
        # to the canonical identity, so the two sources would stay two books
        # and nothing would ever look contested.
        for isbn in ("9780000001009", "9780000001016", "9780000001023", "9780000001030"):
            CatalogueLoader().load(
                engine,
                [
                    openlibrary_view(isbn, title="B", year="1965", pages=100),
                    googlebooks_view(isbn, title="B Other", year="1999", pages=900),
                ],
            )

        assert len(find_contested(engine, minimum_conflicts=1, limit=2)) == 2


class TestResolvingThroughTheTieBreaker:
    @staticmethod
    def _stub_answer(monkeypatch: pytest.MonkeyPatch, observation: Any) -> None:
        async def fake(_extractor: Any, _client: Any, _book: dict[str, Any]) -> Any:
            return observation

        monkeypatch.setattr("pipeline.contested._resolve_one", fake)

    def _contested_pair(self, engine: Engine, isbn: str) -> None:
        CatalogueLoader().load(
            engine,
            [
                openlibrary_view(isbn, title="Contested", year="1965", pages=100),
                googlebooks_view(isbn, title="Contested Other", year="1999", pages=900),
            ],
        )

    def test_an_answer_becomes_a_third_source_not_a_new_book(
        self, engine: Engine, connection: Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure this module was rewritten to prevent.

        The tie-breaker was asked about a *known* book. If its answer is loaded
        on its own identity, the catalogue gains a duplicate instead of a third
        opinion — twenty books, twenty duplicates, and a report saying zero
        errors.
        """
        isbn = "9780000000064"
        self._contested_pair(engine, isbn)
        before = connection.execute(select(books.c.id)).scalars().all()

        self._stub_answer(
            monkeypatch,
            RawBook(
                source=SourceName.GOODREADS,
                source_id="gr-1",
                title="Contested",
                raw_payload={"title": "Contested", "publication_year": "1965"},
            ),
        )
        report = resolve_contested(
            settings_for(str(engine.url)), minimum_conflicts=1, limit=5, engine=engine
        )

        after = connection.execute(select(books.c.id)).scalars().all()
        assert len(after) == len(before), "the tie-breaker created a duplicate book"
        assert report.resolved == 1
        sources = (
            connection.execute(
                select(book_sources.c.source).where(book_sources.c.book_id == before[0])
            )
            .scalars()
            .all()
        )
        assert "goodreads" in sources

    def test_no_answer_is_counted_as_unresolved(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._contested_pair(engine, "9780000000071")
        self._stub_answer(monkeypatch, None)

        report = resolve_contested(
            settings_for(str(engine.url)), minimum_conflicts=1, limit=5, engine=engine
        )

        assert (report.queried, report.resolved, report.unresolved) == (1, 0, 1)

    def test_the_run_is_recorded_either_way(
        self, engine: Engine, connection: Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A run that queried a restricted source must leave a record of having
        # done so, whatever it got back.
        self._contested_pair(engine, "9780000000088")
        self._stub_answer(monkeypatch, None)

        report = resolve_contested(
            settings_for(str(engine.url)), minimum_conflicts=1, limit=5, engine=engine
        )

        row = connection.execute(
            select(ingestion_runs.c.status).where(ingestion_runs.c.id == report.run_id)
        ).scalar_one()
        assert row == "partial_success"

    def test_a_failure_mid_run_still_closes_the_run(
        self, engine: Engine, connection: Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crashed run must be distinguishable from one still going.

        A run row left open is indistinguishable from work in progress, and
        under max_active_runs=1 that is what wedges the next schedule.
        """
        self._contested_pair(engine, "9780000000095")

        async def explode(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("source fell over")

        monkeypatch.setattr("pipeline.contested._resolve_one", explode)

        with pytest.raises(RuntimeError):
            resolve_contested(
                settings_for(str(engine.url)), minimum_conflicts=1, limit=5, engine=engine
            )

        statuses = connection.execute(select(ingestion_runs.c.status)).scalars().all()
        assert statuses == ["failed"]


class TestWhenTheTieBreakerMisbehaves:
    """Every way one book can fail must cost one book, not the run.

    A bounded run over the worst records in the catalogue is only worth
    starting if a single bad answer does not end it.
    """

    def _contested(self, engine: Engine, isbn: str) -> None:
        CatalogueLoader().load(
            engine,
            [
                openlibrary_view(isbn, title="Contested", year="1965", pages=100),
                googlebooks_view(isbn, title="Contested Other", year="1999", pages=900),
            ],
        )

    def test_an_unavailable_source_is_recorded_and_survived(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._contested(engine, "9780000001009")

        async def unavailable(*_a: Any, **_k: Any) -> Any:
            raise GoodreadsUnavailableError("502 from upstream")

        monkeypatch.setattr("pipeline.contested._resolve_one", unavailable)

        report = resolve_contested(
            settings_for(str(engine.url)), minimum_conflicts=1, limit=5, engine=engine
        )

        assert report.unresolved == 1
        assert report.errors
        assert "502" in report.errors[0]

    def test_an_unmappable_answer_is_unresolved(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._contested(engine, "9780000001016")
        # The canonicaliser is stubbed rather than fed a specially broken
        # record: what is under test is what this loop does with a rejection,
        # not which records the canonicaliser rejects.
        monkeypatch.setattr(
            "pipeline.contested.canonicalise",
            lambda _observation: Rejected(
                source=SourceName.GOODREADS,
                source_id="gr-x",
                raw_payload={},
                rejection_code="invalid_record",
                detail="no usable title",
            ),
        )
        monkeypatch.setattr(
            "pipeline.contested._resolve_one",
            _always(
                RawBook(
                    source=SourceName.GOODREADS,
                    source_id="gr-x",
                    title="Contested",
                    raw_payload={"title": "Contested"},
                )
            ),
        )

        report = resolve_contested(
            settings_for(str(engine.url)), minimum_conflicts=1, limit=5, engine=engine
        )

        assert (report.resolved, report.unresolved) == (0, 1)

    def test_an_open_circuit_stops_the_run(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-probing a source that pushed back is what the rules forbid.

        Not a performance choice: once it has refused us, continuing to ask is
        precisely the behaviour the containment rules exist to prevent.
        """
        for isbn in ("9780000001023", "9780000001030"):
            self._contested(engine, isbn)

        opened = {"value": False}

        class Tripping:
            def __init__(self, _settings: Any) -> None:
                pass

            @property
            def circuit_open(self) -> bool:
                return opened["value"]

            @property
            def refused(self) -> bool:
                return opened["value"]

            @property
            def circuit_reason(self) -> str:
                return "access denied: HTTP 403"

            def ensure_accepted(self) -> None:
                return None

            def build_client(self) -> Any:
                class Client:
                    async def aclose(self) -> None:
                        return None

                return Client()

        monkeypatch.setattr("pipeline.contested.GoodreadsExtractor", Tripping)

        async def trip(*_a: Any, **_k: Any) -> Any:
            opened["value"] = True
            return None

        monkeypatch.setattr("pipeline.contested._resolve_one", trip)

        report = resolve_contested(
            settings_for(str(engine.url)), minimum_conflicts=1, limit=5, engine=engine
        )

        assert report.queried == 1, "kept asking a source that had refused"
        assert any("circuit" in error for error in report.errors)

    def test_an_unattachable_answer_does_not_create_a_book(
        self, engine: Engine, connection: Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The guard that turned twenty duplicates into twenty skips.
        isbn = "9780000001047"
        self._contested(engine, isbn)
        monkeypatch.setattr("pipeline.contested._attach_to", lambda *_a, **_k: None)
        monkeypatch.setattr(
            "pipeline.contested._resolve_one",
            _always(
                RawBook(
                    source=SourceName.GOODREADS,
                    source_id="gr-2",
                    title="Contested",
                    raw_payload={"title": "Contested"},
                )
            ),
        )
        before = connection.execute(select(books.c.id)).scalars().all()

        report = resolve_contested(
            settings_for(str(engine.url)), minimum_conflicts=1, limit=5, engine=engine
        )

        assert report.unresolved == 1
        assert connection.execute(select(books.c.id)).scalars().all() == before


class TestAdjudicatedBooksAreNotAskedAgain:
    """Why a tie-breaker run must remember what it already asked.

    Goodreads does not *resolve* a conflict — it adds a third opinion to it,
    and by disagreeing with both documented sources it can raise the conflict
    count. Ranking by conflicts therefore keeps the same handful of books at
    the top of every run, and a bounded run spends its entire budget re-asking
    a restricted source about records it already holds.
    """

    def _contested_pair(self, engine: Engine, isbn: str) -> None:
        CatalogueLoader().load(
            engine,
            [
                openlibrary_view(isbn, title="Contested", year="1965", pages=100),
                googlebooks_view(isbn, title="Contested Other", year="1999", pages=900),
            ],
        )

    def _goodreads_answer(self, engine: Engine, isbn: str) -> None:
        CatalogueLoader().load(
            engine,
            [
                _clean(
                    RawBook(
                        source=SourceName.GOODREADS,
                        source_id=f"gr-{isbn}",
                        title="Contested",
                        isbns=[isbn],
                        raw_payload={"bookId": f"gr-{isbn}", "title": "Contested"},
                    )
                )
            ],
        )

    def test_a_book_with_a_goodreads_answer_is_dropped(self, engine: Engine) -> None:
        isbn = "9780000001115"
        self._contested_pair(engine, isbn)
        assert find_contested(engine, minimum_conflicts=1, limit=10)

        self._goodreads_answer(engine, isbn)

        assert find_contested(engine, minimum_conflicts=1, limit=10) == [], (
            "the tie-breaker would be asked about this book again next run"
        )

    def test_it_still_disagrees_which_is_the_point(self, engine: Engine) -> None:
        # The book is not excluded because it stopped being contested. It is
        # excluded because asking again cannot tell us anything new.
        isbn = "9780000001122"
        self._contested_pair(engine, isbn)
        self._goodreads_answer(engine, isbn)

        assert find_contested(engine, minimum_conflicts=1, limit=10, skip_adjudicated=False)

    def test_books_never_asked_about_are_still_found(self, engine: Engine) -> None:
        # The exclusion must not swallow the backlog it exists to work through.
        answered, unanswered = "9780000001139", "9780000001146"
        self._contested_pair(engine, answered)
        self._contested_pair(engine, unanswered)
        self._goodreads_answer(engine, answered)

        found = find_contested(engine, minimum_conflicts=1, limit=10)

        assert [b["isbn13"] for b in found] == [unanswered]


class TestAnUnacceptedSourceMidRun:
    def test_it_closes_the_run_before_re_raising(
        self, engine: Engine, connection: Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The gate is checked up front, so reaching this means configuration
        # changed under a live run. The run row still has to be closed: a
        # crashed run and one that never started must stay distinguishable.
        CatalogueLoader().load(
            engine,
            [
                openlibrary_view("9780000000255", title="Contested", year="1965", pages=100),
                googlebooks_view("9780000000255", title="Other", year="1999", pages=900),
            ],
        )

        async def refuse(*_args: Any, **_kwargs: Any) -> None:
            raise GoodreadsNotAcceptedError("gate withdrawn mid-run")

        monkeypatch.setattr("pipeline.contested._run", refuse)

        with pytest.raises(GoodreadsNotAcceptedError):
            resolve_contested(
                settings_for(str(engine.url)), minimum_conflicts=1, limit=5, engine=engine
            )

        status = connection.execute(
            select(ingestion_runs.c.status).order_by(ingestion_runs.c.started_at.desc()).limit(1)
        ).scalar_one()
        assert status == "failed"
