"""Discovering candidates from Gutendex.

An alternative to the dump, not a replacement: it indexes Project Gutenberg,
so it is excellent on classics and blind to anything published in the last
century. These tests pin the mapping and the courtesy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pipeline.config import Settings
from pipeline.discover.gutendex_source import _to_candidate, build_manifest_from_gutendex


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


class TestCollectingAManifest:
    """Paging, courtesy, and the manifest write.

    Gutendex is a small volunteer-run service, so the rate limit and the page
    following are as much part of the contract as the mapping is.
    """

    @staticmethod
    def _settings() -> Settings:
        return Settings(  # type: ignore[call-arg]
            database_url="postgresql+psycopg://u:p@localhost/db",
            openlibrary_contact_email="t@example.com",
            gutendex_base_url="https://gutendex.test",
        )

    @staticmethod
    def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> list[float]:
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr("pipeline.discover.gutendex_source.asyncio.sleep", fake_sleep)
        return slept

    @respx.mock
    def test_it_writes_a_manifest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_waiting(monkeypatch)
        respx.get("https://gutendex.test/books/").mock(
            return_value=httpx.Response(200, json={"results": [book(), book(id=1)], "next": None})
        )
        manifest = tmp_path / "candidates.jsonl"

        written = build_manifest_from_gutendex(self._settings(), manifest, max_candidates=10)

        assert written == 2
        lines = manifest.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["title"] == "Pride and Prejudice"

    @respx.mock
    def test_it_follows_pages_until_the_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._no_waiting(monkeypatch)
        # Sequenced rather than two routes: an unconstrained route matches the
        # paged URL too, so the first would answer both requests forever.
        respx.get(url__startswith="https://gutendex.test/books/").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={"results": [book(id=1)], "next": "https://gutendex.test/books/?page=2"},
                ),
                httpx.Response(200, json={"results": [book(id=2)], "next": None}),
            ]
        )

        written = build_manifest_from_gutendex(
            self._settings(), tmp_path / "c.jsonl", max_candidates=10
        )

        assert written == 2

    @respx.mock
    def test_the_cap_stops_paging(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_waiting(monkeypatch)
        route = respx.get(url__startswith="https://gutendex.test/books/").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "results": [book(id=1), book(id=2), book(id=3)],
                        "next": "https://gutendex.test/books/?page=2",
                    },
                ),
                httpx.Response(200, json={"results": [book(id=4)], "next": None}),
            ]
        )

        written = build_manifest_from_gutendex(
            self._settings(), tmp_path / "c.jsonl", max_candidates=2
        )

        assert written == 2
        # A cap that still fetched the next page would be a cap on the file,
        # not on what we ask a volunteer-run service for.
        assert route.call_count == 1

    @respx.mock
    def test_it_waits_between_pages(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        slept = self._no_waiting(monkeypatch)
        respx.get(url__startswith="https://gutendex.test/books/").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={"results": [book(id=1)], "next": "https://gutendex.test/books/?page=2"},
                ),
                httpx.Response(200, json={"results": [book(id=2)], "next": None}),
            ]
        )

        build_manifest_from_gutendex(self._settings(), tmp_path / "c.jsonl", max_candidates=10)

        assert slept == [1.0]

    @respx.mock
    def test_an_unmappable_book_is_skipped_not_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._no_waiting(monkeypatch)
        respx.get("https://gutendex.test/books/").mock(
            return_value=httpx.Response(
                200, json={"results": [{"id": 5}, book(id=6)], "next": None}
            )
        )

        written = build_manifest_from_gutendex(
            self._settings(), tmp_path / "c.jsonl", max_candidates=10
        )

        assert written == 1

    @respx.mock
    def test_an_http_error_is_raised(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_waiting(monkeypatch)
        respx.get("https://gutendex.test/books/").mock(return_value=httpx.Response(503))

        with pytest.raises(httpx.HTTPStatusError):
            build_manifest_from_gutendex(self._settings(), tmp_path / "c.jsonl", max_candidates=10)

    @respx.mock
    def test_a_crash_leaves_no_half_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The manifest is written to a temporary path and moved, so the next
        # task cannot read a partial file as a complete one.
        self._no_waiting(monkeypatch)
        respx.get("https://gutendex.test/books/").mock(return_value=httpx.Response(500))
        manifest = tmp_path / "c.jsonl"

        with pytest.raises(httpx.HTTPStatusError):
            build_manifest_from_gutendex(self._settings(), manifest, max_candidates=10)

        assert not manifest.exists()
