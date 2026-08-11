"""Canonical identity and content hashing.

Identity decides which source records become the same book. It has to be a pure
function of the record's content: the same input on a different day, in a
different process, with a differently-ordered payload must produce the same key,
or re-running the pipeline creates duplicates instead of updating rows.
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pipeline.models.domain import IDENTITY_KEY_PATTERN
from pipeline.transform.identity import (
    FALLBACK_PREFIX,
    ISBN_PREFIX,
    content_hash,
    fallback_identity_key,
    identity_key,
    payload_hash,
)


class TestIdentityKey:
    def test_an_isbn_produces_an_isbn_key(self) -> None:
        key = identity_key(isbn13="9780553380163", title="anything", first_author=None, year=None)

        assert key == f"{ISBN_PREFIX}9780553380163"

    def test_an_isbn_key_ignores_the_other_fields(self) -> None:
        # Two sources describing the same ISBN must agree even when their
        # titles differ in punctuation or their years disagree.
        a = identity_key(
            isbn13="9780553380163",
            title="a brief history of time",
            first_author="stephen hawking",
            year=1988,
        )
        b = identity_key(
            isbn13="9780553380163", title="brief history of time, a", first_author=None, year=1998
        )

        assert a == b

    def test_without_an_isbn_it_falls_back(self) -> None:
        key = identity_key(
            isbn13=None, title="moby dick", first_author="herman melville", year=1851
        )

        assert key.startswith(FALLBACK_PREFIX)
        assert len(key) == len(FALLBACK_PREFIX) + 64

    def test_the_fallback_key_matches_the_documented_recipe(self) -> None:
        direct = fallback_identity_key("moby dick", "herman melville", 1851)
        viaidentity = identity_key(
            isbn13=None, title="moby dick", first_author="herman melville", year=1851
        )

        assert viaidentity == direct

    def test_matches_the_domain_model_pattern(self) -> None:
        # CleanBook validates identity_key against a regex; a key it rejects
        # would fail at the boundary rather than here.
        for key in (
            identity_key(isbn13="9780553380163", title="t", first_author=None, year=None),
            identity_key(isbn13=None, title="t", first_author="a", year=1900),
        ):
            assert re.fullmatch(IDENTITY_KEY_PATTERN, key)


# A normalised title is non-blank by construction; fallback_identity_key
# documents that it refuses anything else, so the properties below generate
# only inputs the function accepts. The refusal has its own test.
normalised_titles = st.text(min_size=1, max_size=40).filter(lambda t: bool(t.strip()))


class TestFallbackDeterminism:
    def test_the_same_inputs_always_give_the_same_key(self) -> None:
        assert fallback_identity_key("moby dick", "herman melville", 1851) == (
            fallback_identity_key("moby dick", "herman melville", 1851)
        )

    def test_a_missing_year_is_distinct_from_a_present_one(self) -> None:
        assert fallback_identity_key("moby dick", "herman melville", None) != (
            fallback_identity_key("moby dick", "herman melville", 1851)
        )

    def test_a_missing_author_is_distinct_from_a_present_one(self) -> None:
        assert fallback_identity_key("moby dick", None, 1851) != (
            fallback_identity_key("moby dick", "herman melville", 1851)
        )

    def test_field_boundaries_cannot_be_forged(self) -> None:
        # Without a separator that cannot appear in a field, "ab" + "c" and
        # "a" + "bc" would hash identically and merge two different books.
        assert fallback_identity_key("ab", "c", None) != fallback_identity_key("a", "bc", None)

    def test_different_titles_give_different_keys(self) -> None:
        assert fallback_identity_key("moby dick", "x", 1851) != (
            fallback_identity_key("moby duck", "x", 1851)
        )

    @given(
        normalised_titles,
        st.one_of(st.none(), st.text(min_size=1, max_size=40)),
        st.one_of(st.none(), st.integers(min_value=1400, max_value=2100)),
    )
    def test_is_stable_across_calls(self, title: str, author: str | None, year: int | None) -> None:
        assert fallback_identity_key(title, author, year) == (
            fallback_identity_key(title, author, year)
        )

    @given(
        normalised_titles,
        st.one_of(st.none(), st.text(min_size=1, max_size=40)),
        st.one_of(st.none(), st.integers(min_value=1400, max_value=2100)),
    )
    def test_always_matches_the_schema_pattern(
        self, title: str, author: str | None, year: int | None
    ) -> None:
        key = fallback_identity_key(title, author, year)

        assert key.startswith(FALLBACK_PREFIX)
        assert len(key) == len(FALLBACK_PREFIX) + 64
        assert all(c in "0123456789abcdef" for c in key[len(FALLBACK_PREFIX) :])

    @pytest.mark.parametrize("blank", ["", " ", "\t\n", "   "])
    def test_a_blank_title_is_refused(self, blank: str) -> None:
        # Every ISBN-less book would otherwise collapse onto one key, merging
        # the entire catalogue into a single row.
        with pytest.raises(ValueError, match="title"):
            fallback_identity_key(blank, None, None)


class TestPayloadHash:
    def test_key_order_does_not_change_the_hash(self) -> None:
        # Providers do not promise stable JSON key order, and a hash that moved
        # with it would report every unchanged record as changed.
        assert payload_hash({"a": 1, "b": 2}) == payload_hash({"b": 2, "a": 1})

    def test_nested_key_order_does_not_change_the_hash(self) -> None:
        assert payload_hash({"o": {"x": 1, "y": 2}}) == payload_hash({"o": {"y": 2, "x": 1}})

    def test_list_order_does_change_the_hash(self) -> None:
        # Order is meaningful in a list: author one is not author two.
        assert payload_hash({"a": [1, 2]}) != payload_hash({"a": [2, 1]})

    def test_different_content_hashes_differently(self) -> None:
        assert payload_hash({"a": 1}) != payload_hash({"a": 2})

    def test_a_missing_key_differs_from_a_null_one(self) -> None:
        assert payload_hash({}) != payload_hash({"a": None})

    def test_unicode_is_handled_consistently(self) -> None:
        assert payload_hash({"t": "Misérables"}) == payload_hash({"t": "Misérables"})

    def test_is_hex_sha256(self) -> None:
        digest = payload_hash({"a": 1})

        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


class TestContentHash:
    def test_the_same_canonical_fields_hash_alike(self) -> None:
        fields = {"title": "Dune", "isbn13": "9780441172719", "published_year": 1965}

        assert content_hash(fields) == content_hash(dict(reversed(list(fields.items()))))

    def test_a_changed_field_changes_the_hash(self) -> None:
        # This is what stops an unchanged record touching books.updated_at.
        base = {"title": "Dune", "published_year": 1965}

        assert content_hash(base) != content_hash({**base, "published_year": 1966})

    def test_none_and_absent_are_the_same_canonical_state(self) -> None:
        # A field the source stopped sending and one it sent as null describe
        # the same book, and must not look like an update.
        assert content_hash({"title": "Dune", "publisher": None}) == (
            content_hash({"title": "Dune"})
        )
