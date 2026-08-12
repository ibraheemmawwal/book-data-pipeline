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
    conflict_count,
    resolve_contested,
)
from pipeline.extract.goodreads import GoodreadsNotAcceptedError


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
