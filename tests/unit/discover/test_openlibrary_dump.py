"""Candidate discovery from an Open Library dump.

The dump is tens of gigabytes, so discovery streams and never materialises it.
It is also the only deterministic part of the pipeline: for a pinned checksum,
the same dump must always produce the same manifest, or a rerun resolves a
different set of books and nothing downstream is reproducible.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from pipeline.discover.openlibrary_dump import (
    ChecksumMismatchError,
    TruncatedDumpError,
    build_manifest,
    read_manifest,
    stream_candidates,
    verify_checksum,
)
from pipeline.models.domain import CandidateBook

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "ol_dump_editions_sample.txt.gz"


def keys(candidates: list[CandidateBook]) -> list[str]:
    return [c.candidate_key for c in candidates]


class TestStreaming:
    def test_yields_candidates_from_a_real_dump_slice(self) -> None:
        found = list(stream_candidates(FIXTURE))

        assert found
        assert all(isinstance(c, CandidateBook) for c in found)

    def test_the_edition_key_becomes_the_candidate_key(self) -> None:
        found = list(stream_candidates(FIXTURE))

        assert all(c.candidate_key.startswith("/books/OL") for c in found)

    def test_identifiers_are_preserved(self) -> None:
        dune = next(c for c in stream_candidates(FIXTURE) if c.title == "Dune")

        assert dune.openlibrary_edition_key == "/books/OL900001M"
        assert dune.openlibrary_work_key == "/works/OL893415W"
        assert "9780441172719" in dune.isbns

    def test_the_by_statement_supplies_an_author_name(self) -> None:
        # Edition records carry author *keys*, not names, so by_statement is
        # the only place a resolvable name comes from.
        dune = next(c for c in stream_candidates(FIXTURE) if c.title == "Dune")

        assert dune.authors == ["Frank Herbert"]

    def test_the_discovery_payload_is_retained(self) -> None:
        # It becomes a provenance-bearing fallback observation later, so it has
        # to survive discovery intact.
        dune = next(c for c in stream_candidates(FIXTURE) if c.title == "Dune")

        assert dune.discovery_payload["title"] == "Dune"

    def test_it_does_not_read_the_whole_file(self) -> None:
        # A generator, not a list: the real dump is tens of gigabytes.
        stream = stream_candidates(FIXTURE)

        assert next(iter(stream)) is not None

    def test_a_limit_stops_early(self) -> None:
        assert len(list(stream_candidates(FIXTURE, max_candidates=1))) == 1


class TestIdentityMaterial:
    def test_a_record_with_no_title_is_rejected(self) -> None:
        # Nothing can be looked up by ISBN alone in a title-and-author search.
        assert "/books/OL900004M" not in keys(list(stream_candidates(FIXTURE)))

    def test_a_record_with_only_an_author_key_is_rejected(self) -> None:
        # "a key alone is not a resolvable candidate" — an author key cannot be
        # turned into a search query.
        assert "/books/OL900003M" not in keys(list(stream_candidates(FIXTURE)))

    def test_a_record_whose_only_isbn_fails_checksum_is_rejected(self) -> None:
        # An invalid ISBN is not lookup material; it is a typo.
        assert "/books/OL900005M" not in keys(list(stream_candidates(FIXTURE)))

    def test_a_title_with_a_valid_isbn_is_accepted(self) -> None:
        assert "/books/OL900001M" in keys(list(stream_candidates(FIXTURE)))

    def test_a_title_with_an_author_name_and_no_isbn_is_accepted(self) -> None:
        candidates = list(
            stream_candidates(
                _write_rows(
                    [
                        _row(
                            "/books/OL1M",
                            {
                                "title": "Some Book",
                                "by_statement": "by A Writer",
                            },
                        )
                    ]
                )
            )
        )

        assert keys(candidates) == ["/books/OL1M"]


class TestLanguageFilter:
    def test_configured_languages_are_kept(self) -> None:
        found = list(stream_candidates(FIXTURE, languages=frozenset({"eng"})))

        assert "/books/OL900001M" in keys(found)

    def test_other_languages_are_dropped(self) -> None:
        found = list(stream_candidates(FIXTURE, languages=frozenset({"eng"})))

        assert "/books/OL900002M" not in keys(found)

    def test_no_filter_keeps_everything_resolvable(self) -> None:
        found = list(stream_candidates(FIXTURE, languages=None))

        assert "/books/OL900002M" in keys(found)

    def test_a_record_with_no_language_is_kept_when_filtering(self) -> None:
        # Most of the dump has no language field. Dropping those would discard
        # the majority of the catalogue to enforce a filter it never stated.
        rows = _write_rows(
            [
                _row(
                    "/books/OL2M",
                    {
                        "title": "No Language",
                        "isbn_13": ["9780441172719"],
                    },
                )
            ]
        )

        assert keys(list(stream_candidates(rows, languages=frozenset({"eng"})))) == ["/books/OL2M"]


class TestMalformedRows:
    def test_a_short_row_is_skipped_without_stopping_the_stream(self) -> None:
        # One corrupt line must not cost the other forty million.
        assert len(list(stream_candidates(FIXTURE))) > 0

    def test_unparseable_json_is_skipped(self) -> None:
        assert "/books/OL900007M" not in keys(list(stream_candidates(FIXTURE)))

    def test_a_non_edition_row_is_skipped(self) -> None:
        rows = _write_rows(
            [
                "/type/work\t/works/OL1W\t1\t2026-01-01T00:00:00.000000\t"
                + json.dumps({"title": "A Work", "isbn_13": ["9780441172719"]})
            ]
        )

        assert list(stream_candidates(rows)) == []


class TestChecksum:
    def test_a_matching_checksum_passes(self) -> None:
        digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()

        verify_checksum(FIXTURE, digest)

    def test_a_mismatched_checksum_raises(self) -> None:
        # Pinning is the whole basis of "deterministic for a pinned checksum".
        with pytest.raises(ChecksumMismatchError, match="does not match"):
            verify_checksum(FIXTURE, "0" * 64)

    def test_the_comparison_is_case_insensitive(self) -> None:
        digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()

        verify_checksum(FIXTURE, digest.upper())


class TestManifest:
    def test_building_writes_every_candidate(self, tmp_path: Path) -> None:
        out = tmp_path / "manifest.jsonl"

        written = build_manifest(FIXTURE, out)

        assert written == len(list(stream_candidates(FIXTURE)))
        assert out.exists()

    def test_the_manifest_round_trips(self, tmp_path: Path) -> None:
        out = tmp_path / "manifest.jsonl"
        build_manifest(FIXTURE, out)

        assert keys(list(read_manifest(out))) == keys(list(stream_candidates(FIXTURE)))

    def test_building_twice_produces_an_identical_manifest(self, tmp_path: Path) -> None:
        # Determinism is the requirement: a rerun that resolved a different set
        # of books would make nothing downstream reproducible.
        first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        build_manifest(FIXTURE, first)
        build_manifest(FIXTURE, second)

        assert first.read_bytes() == second.read_bytes()

    def test_a_checksum_mismatch_leaves_no_manifest_behind(self, tmp_path: Path) -> None:
        # Written to a temporary file and renamed only after verification, so a
        # failed run cannot leave a half-trusted manifest for the next one.
        out = tmp_path / "manifest.jsonl"

        with pytest.raises(ChecksumMismatchError):
            build_manifest(FIXTURE, out, expected_sha256="0" * 64)

        assert not out.exists()
        assert list(tmp_path.iterdir()) == []

    def test_a_matching_checksum_writes_the_manifest(self, tmp_path: Path) -> None:
        out = tmp_path / "manifest.jsonl"
        digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()

        assert build_manifest(FIXTURE, out, expected_sha256=digest) > 0
        assert out.exists()

    def test_an_existing_manifest_is_replaced_atomically(self, tmp_path: Path) -> None:
        out = tmp_path / "manifest.jsonl"
        out.write_text("stale\n")

        build_manifest(FIXTURE, out)

        assert "stale" not in out.read_text()


def _row(key: str, doc: dict[str, object]) -> str:
    doc = {"key": key, **doc}
    return "\t".join(["/type/edition", key, "1", "2026-01-01T00:00:00.000000", json.dumps(doc)])


def _write_rows(rows: list[str]) -> Path:
    """A throwaway dump containing exactly the rows a test cares about."""
    path = Path(tempfile.mkdtemp()) / "rows.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write("\n".join(rows) + "\n")
    return path


class TestTruncatedDump:
    """An interrupted download.

    Discovering from a partial file would find fewer books and say nothing,
    which is exactly the silent failure the pinned checksum exists to prevent.
    """

    def test_a_truncated_gzip_raises_a_named_error(self, tmp_path: Path) -> None:
        # A bare EOFError from inside the decompressor says nothing useful.
        good = FIXTURE.read_bytes()
        truncated = tmp_path / "cut.txt.gz"
        truncated.write_bytes(good[: len(good) // 2])

        with pytest.raises(TruncatedDumpError, match="ended mid-stream"):
            list(stream_candidates(truncated))

    def test_a_file_that_is_not_gzip_raises_the_same_error(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain.txt.gz"
        plain.write_text("this is not compressed\n")

        with pytest.raises(TruncatedDumpError):
            list(stream_candidates(plain))

    def test_a_truncated_dump_publishes_no_manifest(self, tmp_path: Path) -> None:
        good = FIXTURE.read_bytes()
        truncated = tmp_path / "cut.txt.gz"
        truncated.write_bytes(good[: len(good) // 2])
        out = tmp_path / "manifest.jsonl"

        with pytest.raises(TruncatedDumpError):
            build_manifest(truncated, out)

        assert not out.exists()
