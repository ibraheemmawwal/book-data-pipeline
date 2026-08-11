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
from pipeline.transform.canonicalise import (
    canonicalise,
    merge_candidates,
    unify_identity,
)
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


def candidate_for(source: SourceName, **overrides: Any) -> CleanBook:
    """A candidate pinned to SHARED_ISBN."""
    return candidate(source, **overrides)


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
        assert book.authors[0].source_author_id is not None
        assert book.authors[0].source_author_id.startswith("name:")

    def test_name_author_identity_is_stable_across_source_spelling(self) -> None:
        surname_first = clean(authors=[RawAuthor(name="Melville, Herman")])
        natural_order = clean(authors=[RawAuthor(name="Herman Melville")])

        assert surname_first.authors[0].source_author_id == (
            natural_order.authors[0].source_author_id
        )

    def test_invalid_numeric_metadata_becomes_a_rejection(self) -> None:
        result = canonicalise(raw(page_count=-1))

        assert isinstance(result, Rejected)
        assert "page_count" in (result.detail or "")

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
        with_isbn = clean(SourceName.GOOGLEBOOKS, isbns=[SHARED_ISBN])
        without = clean(SourceName.OPENLIBRARY, publisher="Penguin")

        merged = merge_candidates([without, with_isbn])

        assert merged.isbn13 == "9780553380163"
        assert merged.identity_key == "isbn:9780553380163"

    def test_existing_fallback_identity_can_survive_a_metadata_change(self) -> None:
        existing = clean(SourceName.GUTENDEX, title="Moby Dick")
        changed = clean(SourceName.GUTENDEX, title="Moby-Dick", source_id="updated")

        merged = merge_candidates([existing, changed], target_identity_key=existing.identity_key)

        assert merged.identity_key == existing.identity_key

    def test_conflicting_isbns_are_refused(self) -> None:
        first = clean(SourceName.OPENLIBRARY, isbns=["9780553380163"])
        second = clean(SourceName.GOOGLEBOOKS, isbns=["9780306406157"])

        with pytest.raises(ValueError, match="conflicting ISBN"):
            merge_candidates([first, second])

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


class TestIdentityAgreementGuard:
    def test_a_candidate_isbn_conflicting_with_the_retained_row_is_refused(self) -> None:
        # Forcing a candidate onto a different book's ISBN identity would fuse
        # two genuinely different books into a row nothing could separate.
        with pytest.raises(ValueError, match="conflicts with target identity"):
            merge_candidates(
                [candidate(SourceName.GOOGLEBOOKS)],
                target_identity_key="isbn:9780441172719",
            )

    def test_candidates_disagreeing_on_isbn_are_refused(self) -> None:
        one = candidate(SourceName.GOOGLEBOOKS)
        other = clean(SourceName.OPENLIBRARY, isbns=["9780441172719"])

        with pytest.raises(ValueError, match="conflicting ISBN identities"):
            merge_candidates([one, other])

    def test_a_fallback_candidate_is_promoted_to_the_retained_isbn(self) -> None:
        # The stored row is the arbiter after a merge has already happened.
        merged = merge_candidates(
            [candidate(SourceName.GUTENDEX)],
            target_identity_key="isbn:" + SHARED_ISBN,
        )

        assert merged.identity_key == "isbn:" + SHARED_ISBN

    def test_an_author_whose_name_normalises_away_is_skipped(self) -> None:
        # RawBook forbids a blank name, so this is the case where a name is
        # non-blank but has no comparison form. A blank normalised name would
        # collide with every other blank one.
        # "." is non-blank but strips to nothing once punctuation is folded.
        record = raw(authors=[RawAuthor(name="Melville, Herman"), RawAuthor(name=".")])
        result = canonicalise(record)

        assert isinstance(result, CleanBook)
        assert [a.name for a in result.authors] == ["Melville, Herman"]


class TestUnifyIdentity:
    """Several observations of one candidate become one book.

    The resolver knows a Goodreads record and an Open Library record describe
    the same candidate; the load layer only sees independent CleanBooks and
    merges by identity. Without this, 50 candidates produced 92 books in a live
    run — every book two sources resolved existed twice.
    """

    def test_an_isbn_observation_pulls_the_others_onto_its_identity(self) -> None:
        with_isbn = clean(SourceName.OPENLIBRARY, isbns=[SHARED_ISBN])
        without = clean(SourceName.GOODREADS, source_id="gr1")

        unified = unify_identity([with_isbn, without])

        assert {c.identity_key for c in unified} == {"isbn:" + SHARED_ISBN}

    def test_the_isbn_moves_with_the_identity(self) -> None:
        # CleanBook requires the two to agree; leaving isbn13 behind would fail
        # validation the moment the record was rebuilt.
        unified = unify_identity(
            [
                clean(SourceName.OPENLIBRARY, isbns=[SHARED_ISBN]),
                clean(SourceName.GOODREADS, source_id="gr1"),
            ]
        )

        assert all(c.isbn13 == SHARED_ISBN for c in unified)

    def test_each_observation_keeps_its_own_provenance(self) -> None:
        # One book, but still one book_sources row per source.
        unified = unify_identity(
            [
                clean(SourceName.OPENLIBRARY, isbns=[SHARED_ISBN]),
                clean(SourceName.GOODREADS, source_id="gr1"),
            ]
        )

        assert {c.source for c in unified} == {
            SourceName.OPENLIBRARY,
            SourceName.GOODREADS,
        }

    def test_without_an_isbn_the_most_complete_record_sets_the_identity(self) -> None:
        rich = clean(SourceName.GOODREADS, source_id="gr1", description="x", publisher="y")
        sparse = clean(SourceName.GUTENDEX, source_id="1")

        unified = unify_identity([rich, sparse])

        assert len({c.identity_key for c in unified}) == 1

    def test_conflicting_isbns_are_refused(self) -> None:
        # Two different books behind one candidate. Fusing them would create a
        # row nothing could separate again.
        a = clean(SourceName.OPENLIBRARY, isbns=[SHARED_ISBN])
        b = clean(SourceName.GOODREADS, source_id="gr1", isbns=["9780441172719"])

        with pytest.raises(ValueError, match="disagree on ISBN"):
            unify_identity([a, b])

    def test_a_single_observation_is_returned_unchanged(self) -> None:
        only = clean(SourceName.GOODREADS, source_id="gr1")

        assert unify_identity([only])[0].identity_key == only.identity_key

    def test_an_empty_list_is_empty(self) -> None:
        assert unify_identity([]) == []
