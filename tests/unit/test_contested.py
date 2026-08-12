"""Re-resolving contested books through a tie-breaker.

The justification for using a restricted source here is that the set is small
and chosen for a reason. These tests pin both halves of that: which books
qualify, and that the bound is real rather than aspirational.
"""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.config import Settings
from pipeline.contested import (
    COMPARABLE_KEYS,
    ContestedReport,
    _attach_to,
    conflict_count,
    resolve_contested,
)
from pipeline.extract.goodreads import GoodreadsNotAcceptedError
from pipeline.models.domain import CleanBook, SourceName


class TestConflictCounting:
    def test_agreeing_sources_produce_no_conflicts(self) -> None:
        assert conflict_count([{"title": "Dune"}, {"title": "Dune"}]) == 0

    def test_a_single_source_cannot_conflict(self) -> None:
        assert conflict_count([{"title": "Dune"}]) == 0

    def test_differing_titles_count_once(self) -> None:
        assert conflict_count([{"title": "Dune"}, {"title": "Dune: A Study Guide"}]) == 1

    def test_case_differences_are_not_conflicts(self) -> None:
        assert conflict_count([{"title": "DUNE"}, {"title": "dune"}]) == 0

    def test_several_fields_accumulate(self) -> None:
        count = conflict_count(
            [
                {"title": "A", "publisher": "Ace", "number_of_pages": 100},
                {"title": "B", "publisher": "Penguin", "pageCount": 200},
            ]
        )

        assert count == 3

    def test_nested_payloads_are_compared(self) -> None:
        # Google Books nests under volumeInfo; a flat read would report
        # agreement and quietly exclude the book from re-resolution.
        count = conflict_count(
            [{"first_publish_year": 1965}, {"volumeInfo": {"publishedDate": "1990"}}]
        )

        assert count == 1

    def test_a_missing_value_is_not_a_conflict(self) -> None:
        # Silence is not disagreement.
        assert conflict_count([{"title": "Dune"}, {}]) == 0

    def test_a_non_dict_payload_is_ignored(self) -> None:
        assert conflict_count([{"title": "Dune"}, ["not", "a", "dict"]]) == 0

    def test_the_field_list_matches_what_the_api_compares(self) -> None:
        """The pipeline and the API must mean the same thing by 'contested'.

        If these drifted, a book the catalogue shows as contested would not be
        one this module re-resolves, and nobody would notice.
        """
        assert set(COMPARABLE_KEYS) == {
            "title",
            "published_year",
            "publisher",
            "page_count",
        }


class TestGates:
    def test_it_refuses_when_the_source_is_not_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A targeted run is not a reason to skip the acknowledgement.

        Both gates exist because the source's terms restrict automated
        collection, and that does not change with volume.
        """
        settings = Settings(  # type: ignore[call-arg]
            database_url="postgresql+psycopg://u:p@localhost/db",
            openlibrary_contact_email="t@example.com",
            goodreads_enabled=True,
            goodreads_unofficial_source_accepted=False,
        )

        class FakeEngine:
            def begin(self) -> Any:
                raise AssertionError("must refuse before opening a run")

        monkeypatch.setattr(
            "pipeline.contested.find_contested",
            lambda *_a, **_k: [
                {"id": 1, "title": "Dune", "isbn13": None, "conflicts": 3, "sources": ["a", "b"]}
            ],
        )

        with pytest.raises(GoodreadsNotAcceptedError):
            resolve_contested(settings, engine=FakeEngine())  # type: ignore[arg-type]

    def test_no_contested_books_makes_no_requests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Gates set: the point of this test is that having nothing to do costs
        # nothing, not that an unaccepted source is refused.
        settings = Settings(  # type: ignore[call-arg]
            database_url="postgresql+psycopg://u:p@localhost/db",
            openlibrary_contact_email="t@example.com",
            goodreads_enabled=True,
            goodreads_unofficial_source_accepted=True,
        )
        monkeypatch.setattr("pipeline.contested.find_contested", lambda *_a, **_k: [])

        class FakeEngine:
            def begin(self) -> Any:
                raise AssertionError("must not open a run with nothing to do")

        report = resolve_contested(settings, engine=FakeEngine())  # type: ignore[arg-type]

        assert report.contested == 0
        assert report.queried == 0


class TestReport:
    def test_it_starts_empty(self) -> None:
        report = ContestedReport()

        assert (report.queried, report.resolved, report.loaded) == (0, 0, 0)


class TestAttachingToTheKnownBook:
    """Re-keying an observation onto the book it was fetched for.

    This is the difference between enriching a record and creating one. The
    tie-breaker is asked about a *known* book; if its answer carries no ISBN it
    derives a fresh fallback identity, the loader sees an unfamiliar record,
    and the contested book quietly gains a duplicate instead of a third source.
    That happened on the first live run: 20 books queried, 20 duplicates
    created, and nothing failed.
    """

    def _clean(self, **overrides: Any) -> Any:
        base: dict[str, Any] = {
            "source": SourceName.GOODREADS,
            "source_id": "1",
            "identity_key": "fallback:" + "a" * 64,
            "title": "Dune",
            "normalised_title": "dune",
            "raw_payload": {},
        }
        return CleanBook(**{**base, **overrides})

    def test_a_fallback_identity_is_carried_across(self) -> None:
        target = {"identity_key": "fallback:" + "b" * 64, "isbn13": None, "title": "Dune"}

        attached = _attach_to(self._clean(), target)

        assert attached is not None
        assert attached.identity_key == target["identity_key"]

    def test_an_isbn_identity_carries_its_isbn_with_it(self) -> None:
        # They must move together: an identity naming one ISBN while the record
        # carries another merges the wrong books.
        target = {
            "identity_key": "isbn:9780553380163",
            "isbn13": "9780553380163",
            "title": "Dune",
        }

        attached = _attach_to(self._clean(), target)

        assert attached is not None
        assert attached.isbn13 == "9780553380163"
        assert attached.identity_key == "isbn:9780553380163"

    def test_an_irreconcilable_pair_is_refused_not_guessed(self) -> None:
        target = {"identity_key": "isbn:9780553380163", "isbn13": None, "title": "Dune"}

        assert _attach_to(self._clean(), target) is None

    def test_the_observation_keeps_its_own_source(self) -> None:
        # Re-keying changes which book it describes, not who reported it —
        # provenance would be a lie otherwise.
        target = {"identity_key": "fallback:" + "b" * 64, "isbn13": None, "title": "Dune"}

        attached = _attach_to(self._clean(), target)

        assert attached is not None
        assert attached.source is SourceName.GOODREADS


class TestIdentitiesThatCannotBeAttached:
    """The pairs that must be refused rather than reconciled.

    identity_key and isbn13 have to move together. A fallback identity holding
    an ISBN, or an ISBN identity naming a different one, merges two different
    books — and no later query can tell that it happened.
    """

    @staticmethod
    def _observation() -> Any:
        return CleanBook(
            source=SourceName.GOODREADS,
            source_id="1",
            identity_key="fallback:" + "a" * 64,
            title="Contested",
            normalised_title="contested",
            raw_payload={},
        )

    def test_an_isbn_identity_naming_a_different_isbn_is_refused(self) -> None:
        book = {
            "identity_key": "isbn:9780441172719",
            "isbn13": "9780553293357",
            "title": "Mismatched",
        }

        assert _attach_to(self._observation(), book) is None

    def test_a_fallback_identity_holding_an_isbn_is_refused(self) -> None:
        book = {"identity_key": "fallback:abc123", "isbn13": "9780441172719", "title": "Wrong"}

        assert _attach_to(self._observation(), book) is None

    def test_an_unrecognised_identity_scheme_is_refused(self) -> None:
        # Not a guess about what a future scheme might mean: an identity this
        # code does not understand is one it must not act on.
        book = {"identity_key": "oclc:12345", "isbn13": None, "title": "Unknown scheme"}

        assert _attach_to(self._observation(), book) is None
