"""Canonicalisation: RawBook to CleanBook, and merging across sources.

The merge rules decide what the catalogue says when three providers disagree.
They must be deterministic — the same set of records in any order must produce
the same book — because the load stage recomputes canonical fields on every
ingest and a wobbling result would rewrite rows that did not change.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from pipeline.extract.base import Rejected
from pipeline.models.domain import CleanBook, RawAuthor, RawBook, SourceName
from pipeline.transform.canonicalise import canonicalise, merge_candidates
from pipeline.transform.isbn import isbn13_check_digit


def raw(source: SourceName = SourceName.GUTENDEX, **overrides: Any) -> RawBook:
    base: dict[str, Any] = {
        "source": source,
        "source_id": "1",
        "title": "Moby Dick",
        "raw_payload": {"id": 1},
    }
    return RawBook(**(base | overrides))


def clean(source: SourceName = SourceName.GUTENDEX, **overrides: Any) -> CleanBook:
    result = canonicalise(raw(source, **overrides))
    assert isinstance(result, CleanBook), result
    return result


SHARED_ISBN = "9780553380163"


def candidate(source: SourceName, **overrides: Any) -> CleanBook:
    """A merge candidate pinned to one canonical identity.

    Merging is only defined within a single identity, so the shared ISBN is
    what lets these records disagree about titles and still be the same book —
    which is exactly the case the merge rules exist for.
    """
    return clean(source, isbns=[SHARED_ISBN], source_id=source.value, **overrides)


class TestCanonicalise:
    def test_maps_a_minimal_record(self) -> None:
        book = clean()

        assert book.title == "Moby Dick"
        assert book.normalised_title == "moby dick"
        assert book.isbn13 is None
        assert book.identity_key.startswith("fallback:")

    def test_display_values_are_preserved_alongside_comparison_forms(self) -> None:
        book = clean(title="  The   GREAT Gatsby ")

        # The reader sees the source's text; matching uses the folded form.
        assert book.title == "The   GREAT Gatsby"
        assert book.normalised_title == "the great gatsby"

    def test_a_valid_isbn_becomes_the_identity(self) -> None:
        book = clean(isbns=["0553380168"])

        assert book.isbn13 == "9780553380163"
        assert book.identity_key == "isbn:9780553380163"

    def test_an_invalid_isbn_is_dropped_not_carried(self) -> None:
        book = clean(isbns=["9780553380164"])

        assert book.isbn13 is None
        assert book.identity_key.startswith("fallback:")

    def test_a_work_level_isbn_list_falls_back(self) -> None:
        book = clean(
            isbns=[
                body + str(isbn13_check_digit(body))
                for body in (f"97805533{i:04d}" for i in range(40))
            ]
        )

        assert book.isbn13 is None
        assert book.identity_key.startswith("fallback:")

    def test_the_year_is_parsed_from_source_text(self) -> None:
        assert clean(published="c1997").published_year == 1997

    def test_an_unusable_year_becomes_none_rather_than_rejecting_the_book(self) -> None:
        book = clean(published="forthcoming")

        assert book.published_year is None
        assert book.title == "Moby Dick"

    def test_language_is_normalised_to_639_3(self) -> None:
        assert clean(languages=["en"]).language == "eng"

    def test_an_ambiguous_language_list_resolves_to_none(self) -> None:
        assert clean(languages=["eng", "cze", "fre"]).language is None

    def test_the_first_author_drives_the_fallback_identity(self) -> None:
        book = clean(authors=[RawAuthor(name="Melville, Herman")])

        assert book.normalised_first_author == "herman melville"

    def test_surname_first_and_natural_order_produce_one_identity(self) -> None:
        # Gutendex and Google Books write the same person differently.
        a = clean(SourceName.GUTENDEX, authors=[RawAuthor(name="Melville, Herman")])
        b = clean(SourceName.GOOGLEBOOKS, authors=[RawAuthor(name="Herman Melville")])

        assert a.identity_key == b.identity_key

    def test_authors_are_carried_through_with_their_lifespans(self) -> None:
        book = clean(authors=[RawAuthor(name="Homer", birth_year=-750, death_year=-650)])

        assert book.authors[0].birth_year == -750

    def test_provenance_survives(self) -> None:
        book = clean(SourceName.OPENLIBRARY, source_id="/works/OL1W")

        assert book.source is SourceName.OPENLIBRARY
        assert book.source_id == "/works/OL1W"
        assert book.raw_payload == {"id": 1}

    def test_a_title_that_normalises_to_nothing_is_rejected(self) -> None:
        # RawBook already forbids a blank title, so this guard is defence in
        # depth: canonicalise must not assume its input came through that
        # validator. Constructed unvalidated to reach the branch at all.
        unvalidated = RawBook.model_construct(
            source=SourceName.GUTENDEX,
            source_id="1",
            title="   ",
            authors=[],
            subjects=[],
            isbns=[],
            languages=[],
            raw_payload={},
        )

        result = canonicalise(unvalidated)

        assert isinstance(result, Rejected)
        assert result.rejection_code == "unidentifiable"

    def test_canonicalisation_is_deterministic(self) -> None:
        assert clean(isbns=["0553380168"]).identity_key == (
            clean(isbns=["0553380168"]).identity_key
        )


class TestMergePrecedence:
    def test_a_single_candidate_is_returned_as_is(self) -> None:
        only = candidate(SourceName.GUTENDEX)

        assert merge_candidates([only]).title == only.title

    def test_the_more_complete_record_wins(self) -> None:
        sparse = candidate(SourceName.GUTENDEX, title="Dune")
        rich = candidate(
            SourceName.GOOGLEBOOKS,
            title="Dune",
            publisher="Ace",
            page_count=412,
            published="1965",
            description="Desert planet.",
        )

        merged = merge_candidates([sparse, rich])

        assert merged.publisher == "Ace"
        assert merged.page_count == 412
        assert merged.published_year == 1965

    def test_source_priority_breaks_a_completeness_tie(self) -> None:
        # openlibrary > googlebooks > gutendex when both look equally complete.
        gut = candidate(SourceName.GUTENDEX, title="Gutendex title")
        ol = candidate(SourceName.OPENLIBRARY, title="Open Library title")

        assert merge_candidates([gut, ol]).title == "Open Library title"
        assert merge_candidates([ol, gut]).title == "Open Library title"

    def test_googlebooks_outranks_gutendex(self) -> None:
        gut = candidate(SourceName.GUTENDEX, title="Gutendex title")
        gb = candidate(SourceName.GOOGLEBOOKS, title="Google title")

        assert merge_candidates([gut, gb]).title == "Google title"

    def test_a_null_field_never_overwrites_a_present_one(self) -> None:
        # Priority decides between two answers, not between an answer and none.
        rich = candidate(SourceName.GUTENDEX, description="A whale of a tale.", download_count=1000)
        sparse = candidate(SourceName.OPENLIBRARY, publisher="Penguin")

        merged = merge_candidates([rich, sparse])

        assert merged.description == "A whale of a tale."
        assert merged.download_count == 1000
        assert merged.publisher == "Penguin"

    def test_merging_is_order_independent(self) -> None:
        a = candidate(SourceName.GUTENDEX, description="From Gutendex", download_count=7)
        b = candidate(SourceName.OPENLIBRARY, publisher="Penguin", published="1851")
        c = candidate(SourceName.GOOGLEBOOKS, page_count=635, publisher="Bantam")

        forward = merge_candidates([a, b, c])
        backward = merge_candidates([c, b, a])

        assert forward.model_dump() == backward.model_dump()

    def test_provider_timestamps_are_never_used_to_decide(self) -> None:
        # The TRD is explicit: source_updated semantics differ per provider and
        # Gutendex has none at all, so a newer timestamp must not win.
        old_but_richer = candidate(
            SourceName.OPENLIBRARY,
            publisher="Penguin",
            page_count=100,
            source_updated=datetime(2000, 1, 1, tzinfo=UTC),
        )
        new_but_sparse = candidate(
            SourceName.GUTENDEX, source_updated=datetime(2026, 1, 1, tzinfo=UTC)
        )

        merged = merge_candidates([old_but_richer, new_but_sparse])

        assert merged.publisher == "Penguin"

    def test_an_isbn_from_any_candidate_is_kept(self) -> None:
        with_isbn = candidate(SourceName.GOOGLEBOOKS)
        # Same identity, but this record never carried the ISBN itself.
        without = candidate(SourceName.OPENLIBRARY, publisher="Penguin").model_copy(
            update={"isbn13": None}
        )

        merged = merge_candidates([without, with_isbn])

        assert merged.isbn13 == "9780553380163"

    def test_authors_and_subjects_are_taken_from_the_winning_record(self) -> None:
        rich = candidate(
            SourceName.GUTENDEX,
            authors=[RawAuthor(name="Melville, Herman", birth_year=1819)],
            subjects=["Whaling"],
            description="x",
            download_count=1,
        )
        sparse = candidate(SourceName.OPENLIBRARY)

        merged = merge_candidates([rich, sparse])

        assert merged.authors[0].birth_year == 1819
        assert merged.subjects == ["Whaling"]

    def test_rejects_an_empty_candidate_list(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            merge_candidates([])

    def test_candidates_with_different_identities_are_refused(self) -> None:
        # Merging across identities would silently fuse two different books.
        a = clean(title="Moby Dick")
        b = clean(title="Pride and Prejudice")  # different fallback identity

        with pytest.raises(ValueError, match="identity"):
            merge_candidates([a, b])
