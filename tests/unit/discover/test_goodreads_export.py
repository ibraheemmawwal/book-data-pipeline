"""Candidates from a Goodreads export gathered elsewhere.

The export is discovery, not a live source: it says these books exist and here
is enough to look them up, which is the claim the Open Library dump makes. What
these tests pin is the translation into the shape the Goodreads mapper already
reads — one reader, one contract, so a record replayed by the loader is the
same whether it came from this file or a live search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.discover.goodreads_export import (
    build_manifest,
    inspect_export,
    stream_candidates,
    to_goodreads_payload,
)
from pipeline.extract import map_payload
from pipeline.models.domain import CleanBook, SourceName
from pipeline.transform import canonicalise


def record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": "To Kill a Mockingbird",
        "author": ["Harper Lee"],
        "rating": 4.26,
        "ratingCount": 6674605,
        "coverUrl": "https://i.gr-assets.com/images/S/x/2657._SY75_.jpg",
        "goodreadsId": "2657",
        "goodreadsUrl": "https://www.goodreads.com/book/show/2657",
        "position": 1,
        "sourceFile": "standalone.json",
    }
    return {**base, **overrides}


def export(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    path = tmp_path / "export.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


class TestTheTranslation:
    def test_it_produces_something_the_goodreads_mapper_reads(self) -> None:
        """The point of translating rather than teaching a second dialect.

        Whatever the loader replays years from now has to be one shape.
        """
        payload = to_goodreads_payload(record())
        assert payload is not None

        mapped = map_payload(SourceName.GOODREADS, payload)
        cleaned = canonicalise(mapped)

        assert isinstance(cleaned, CleanBook)
        assert cleaned.title == "To Kill a Mockingbird"
        assert cleaned.source_id == "2657"
        assert cleaned.goodreads_average_rating is not None

    def test_every_credited_author_is_kept(self) -> None:
        # The search card this pipeline scrapes carries one author; the export
        # carries the list, which is most of why it is worth having.
        payload = to_goodreads_payload(record(author=["Terry Pratchett", "Neil Gaiman"]))

        assert payload is not None
        assert [a["name"] for a in payload["authors"]] == ["Terry Pratchett", "Neil Gaiman"]

    def test_a_record_with_no_id_is_refused(self) -> None:
        # Not a thin record — not a record. It cannot become provenance,
        # because provenance is keyed on the source's own identifier.
        assert to_goodreads_payload(record(goodreadsId=None)) is None

    def test_a_record_with_no_title_is_refused(self) -> None:
        assert to_goodreads_payload(record(title="   ")) is None

    def test_where_it_came_from_is_recorded(self) -> None:
        # A record replayed later should say whether a file supplied it.
        payload = to_goodreads_payload(record())

        assert payload is not None
        assert payload["_export"]["source_file"] == "standalone.json"


class TestStreaming:
    def test_it_yields_a_candidate_per_record(self, tmp_path: Path) -> None:
        path = export(tmp_path, [record(goodreadsId="1"), record(goodreadsId="2")])

        candidates = list(stream_candidates(path))

        assert [c.candidate_key for c in candidates] == ["goodreads:1", "goodreads:2"]
        assert all(c.discovery_source is SourceName.GOODREADS for c in candidates)

    def test_the_cap_is_honoured(self, tmp_path: Path) -> None:
        path = export(tmp_path, [record(goodreadsId=str(n)) for n in range(10)])

        assert len(list(stream_candidates(path, max_candidates=3))) == 3

    def test_it_resumes_where_the_last_run_stopped(self, tmp_path: Path) -> None:
        """Why resuming counts records read, not candidates emitted.

        A 32,000-record export is many runs' work. Starting over each night
        would re-resolve the beginning and never reach the end — which is
        exactly what the dump reader was doing before it learned to resume.
        """
        path = export(tmp_path, [record(goodreadsId=str(n)) for n in range(6)])

        resumed = list(stream_candidates(path, start_index=4))

        assert [c.candidate_key for c in resumed] == ["goodreads:4", "goodreads:5"]

    def test_unusable_records_are_skipped_not_fatal(self, tmp_path: Path) -> None:
        path = export(
            tmp_path, [record(goodreadsId="1"), {"nonsense": True}, record(goodreadsId="3")]
        )

        assert len(list(stream_candidates(path))) == 2

    def test_a_wrapped_list_is_understood(self, tmp_path: Path) -> None:
        # Some exports wrap the array; the key is not guessed at by name.
        path = tmp_path / "wrapped.json"
        path.write_text(json.dumps({"books": [record()]}), encoding="utf-8")

        assert len(list(stream_candidates(path))) == 1

    def test_a_file_that_is_not_a_list_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"title": "not a list"}), encoding="utf-8")

        assert list(stream_candidates(path)) == []


class TestTheManifest:
    def test_it_writes_the_same_format_the_dump_does(self, tmp_path: Path) -> None:
        # Everything downstream must be unable to tell which source discovered
        # a candidate.
        path = export(tmp_path, [record(goodreadsId="1"), record(goodreadsId="2")])
        manifest = tmp_path / "candidates.jsonl"

        assert build_manifest(path, manifest) == 2

        lines = manifest.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["candidate_key"] == "goodreads:1"

    def test_a_crash_leaves_no_half_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "candidates.jsonl"
        missing = tmp_path / "absent.json"

        with pytest.raises(FileNotFoundError, match="absent"):
            build_manifest(missing, manifest)

        assert not manifest.exists()


class TestMalformedExports:
    """A file that is not an export at all.

    Worth distinguishing from an export with bad rows: a record the reader
    cannot use is skipped and counted, but a file whose top level is not a list
    of records means the wrong path was configured, and continuing silently
    would produce an empty run that looks like a finished one.
    """

    def test_a_scalar_file_is_an_error_not_an_empty_run(self, tmp_path: Path) -> None:
        path = tmp_path / "scalar.json"
        path.write_text(json.dumps("not an export"), encoding="utf-8")

        with pytest.raises(ValueError, match="expected a list"):
            list(stream_candidates(path))

    def test_a_number_is_refused_the_same_way(self, tmp_path: Path) -> None:
        path = tmp_path / "number.json"
        path.write_text("42", encoding="utf-8")

        with pytest.raises(ValueError, match="expected a list"):
            list(stream_candidates(path))

    def test_non_object_entries_are_skipped(self, tmp_path: Path) -> None:
        # A stray string in the array is a bad row, not a bad file.
        path = export(tmp_path, [record(goodreadsId="1")])
        path.write_text(
            json.dumps([record(goodreadsId="1"), "stray", None, record(goodreadsId="2")]),
            encoding="utf-8",
        )

        assert [c.candidate_key for c in stream_candidates(path)] == [
            "goodreads:1",
            "goodreads:2",
        ]


class TestAuthorParsing:
    def test_blank_and_non_string_authors_are_dropped(self) -> None:
        payload = to_goodreads_payload(record(author=["Harper Lee", "  ", None, 7]))

        assert payload is not None
        assert [a["name"] for a in payload["authors"]] == ["Harper Lee"]

    def test_a_repeated_author_is_credited_once(self) -> None:
        payload = to_goodreads_payload(record(author=["Ann Leckie", "Ann Leckie"]))

        assert payload is not None
        assert [a["name"] for a in payload["authors"]] == ["Ann Leckie"]

    def test_a_bare_string_author_is_accepted(self) -> None:
        # Not every export wraps a single author in a list.
        payload = to_goodreads_payload(record(author="Ursula K. Le Guin"))

        assert payload is not None
        assert [a["name"] for a in payload["authors"]] == ["Ursula K. Le Guin"]

    def test_no_author_still_yields_a_usable_record(self) -> None:
        payload = to_goodreads_payload(record(author=[]))

        assert payload is not None
        assert payload["author"] == {}


class TestInspectingAnExportBeforeUsingIt:
    """Whether a file can be ingested, answered in a second.

    A file with the wrong shape produces an empty manifest, which produces a
    run that resolves nothing and reports success — an hour spent to learn that
    a path was wrong. The report names the count, the missing field and the
    keys the file actually has, which is what distinguishes "wrong file" from
    "right file, changed format".
    """

    def test_a_good_export_is_compatible(self, tmp_path: Path) -> None:
        report = inspect_export(export(tmp_path, [record(), record(goodreadsId="2")]))

        assert report.compatible
        assert (report.total, report.usable) == (2, 2)
        assert "2 of 2 records usable" in report.explain()

    def test_a_missing_file_says_so(self, tmp_path: Path) -> None:
        report = inspect_export(tmp_path / "absent.json")

        assert not report.compatible
        assert "not found" in report.explain()

    def test_a_file_that_is_not_json_says_so(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json at all", encoding="utf-8")

        assert "not valid JSON" in inspect_export(path).explain()

    def test_a_file_of_the_wrong_shape_says_so(self, tmp_path: Path) -> None:
        path = tmp_path / "scalar.json"
        path.write_text('"a string"', encoding="utf-8")

        report = inspect_export(path)

        assert not report.compatible
        assert "expected a list" in report.explain()

    def test_records_missing_the_id_are_named_and_counted(self, tmp_path: Path) -> None:
        # Which field is missing is the difference between a wrong file and a
        # format change, so the report says rather than reporting "0 usable".
        path = export(tmp_path, [record(goodreadsId=None), record(goodreadsId=None)])

        report = inspect_export(path)

        assert not report.compatible
        assert report.missing_id == 2
        assert "lack goodreadsId" in report.explain()

    def test_the_keys_it_did_find_are_reported(self, tmp_path: Path) -> None:
        # So an operator can see it pointed at some other kind of file.
        path = tmp_path / "other.json"
        path.write_text(json.dumps([{"isbn": "x", "name": "y"}]), encoding="utf-8")

        report = inspect_export(path)

        assert not report.compatible
        assert "isbn" in report.explain()

    def test_a_partly_usable_file_is_still_compatible(self, tmp_path: Path) -> None:
        # A few bad rows are not a reason to refuse 32,000 good ones; they are
        # skipped and counted.
        path = export(tmp_path, [record(), {"junk": True}, record(goodreadsId="3")])

        report = inspect_export(path)

        assert report.compatible
        assert (report.total, report.usable) == (3, 2)
        assert "1 skipped" in report.explain()
