"""Discovering candidates from Gutendex.

An alternative to the dump, not a replacement: it indexes Project Gutenberg,
so it is excellent on classics and blind to anything published in the last
century. These tests pin the mapping and the courtesy.
"""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.discover.gutendex_source import _to_candidate


def book(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": 1342,
        "title": "Pride and Prejudice",
        "authors": [{"name": "Austen, Jane", "birth_year": 1775}],
        "languages": ["en"],
    }
    return {**base, **overrides}


class TestMapping:
    def test_a_complete_book_becomes_a_candidate(self) -> None:
        candidate = _to_candidate(book())

        assert candidate is not None
        assert candidate.title == "Pride and Prejudice"
        assert candidate.authors == ["Austen, Jane"]

    def test_the_key_is_namespaced(self) -> None:
        # Gutendex ids are small integers and would collide with anything else
        # keyed numerically; the prefix says where the candidate came from.
        candidate = _to_candidate(book())

        assert candidate is not None
        assert candidate.candidate_key == "gutendex:1342"

    def test_the_payload_is_retained(self) -> None:
        # So the resolver can promote it to a provenance-bearing observation
        # without spending a second request on the same book.
        candidate = _to_candidate(book())

        assert candidate is not None
        assert candidate.discovery_payload["id"] == 1342

    def test_languages_are_carried(self) -> None:
        candidate = _to_candidate(book())

        assert candidate is not None
        assert candidate.languages == ["en"]


class TestRefusal:
    def test_a_book_with_no_author_is_refused(self) -> None:
        # Same bar as the dump: a title alone matches thousands of books, and a
        # record we cannot resolve is a request we should never send.
        assert _to_candidate(book(authors=[])) is None

    @pytest.mark.parametrize("title", ["", "   ", None])
    def test_a_missing_title_is_refused(self, title: Any) -> None:
        assert _to_candidate(book(title=title)) is None

    def test_a_missing_id_is_refused(self) -> None:
        assert _to_candidate(book(id=None)) is None

    def test_malformed_authors_are_ignored(self) -> None:
        assert _to_candidate(book(authors=[{"no_name": True}])) is None


class TestBounds:
    def test_authors_are_capped(self) -> None:
        # A search query built from twenty names matches nothing.
        many = [{"name": f"Author {n}"} for n in range(20)]

        candidate = _to_candidate(book(authors=many))

        assert candidate is not None
        assert len(candidate.authors) == 5
