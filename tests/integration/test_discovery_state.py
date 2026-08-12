"""Discovery resuming across runs.

Without this a scheduled pipeline is a repeated no-op: the same candidates
every night, already held, and the rest of the dump never reached.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from sqlalchemy import Connection

from pipeline.discover import build_manifest
from pipeline.discover.state import dump_key, read_position, reset_position, save_position

pytestmark = pytest.mark.integration


def write_dump(path: Path, count: int, *, start: int = 0) -> Path:
    """A dump of `count` synthetic editions.

    Each carries a by_statement, because discovery requires something to look
    the book up *with* — a title alone matches thousands. Author keys do not
    count: the dump stores keys, not names, and a key cannot be searched on.
    """
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index in range(start, start + count):
            document = {
                # The key lives inside the JSON as well as in the TSV column;
                # a real record carries both, and discovery reads the former.
                "key": f"/books/OL{index}M",
                "title": f"Book {index:05d}",
                "by_statement": f"by Author {index:05d}",
            }
            handle.write(
                f"/type/edition\t/books/OL{index}M\t1\t2026-01-01T00:00:00\t"
                f"{json.dumps(document)}\n"
            )
    return path


class TestPosition:
    def test_an_unseen_dump_starts_at_the_beginning(self, connection: Connection) -> None:
        position = read_position(connection, "never-seen")

        assert position.line_offset == 0
        assert not position.exhausted

    def test_a_saved_position_is_read_back(self, connection: Connection) -> None:
        save_position(connection, "d", line_offset=120, candidates_emitted=40, exhausted=False)

        assert read_position(connection, "d").line_offset == 120

    def test_emitted_counts_accumulate_across_runs(self, connection: Connection) -> None:
        # The offset is where we are; the count is how much we have produced.
        # Overwriting the count would lose the total.
        save_position(connection, "d", line_offset=10, candidates_emitted=5, exhausted=False)
        save_position(connection, "d", line_offset=20, candidates_emitted=7, exhausted=False)

        assert read_position(connection, "d").candidates_emitted == 12

    def test_a_reset_starts_the_dump_again(self, connection: Connection) -> None:
        save_position(connection, "d", line_offset=99, candidates_emitted=1, exhausted=True)

        reset_position(connection, "d")

        assert read_position(connection, "d").line_offset == 0

    def test_the_key_changes_when_the_dump_does(self, tmp_path: Path) -> None:
        """A refreshed dump must not resume into unrelated data.

        The published file is always named 'latest' and its contents change
        monthly, so name alone would resume at a line describing a different
        book.
        """
        small = write_dump(tmp_path / "d.txt.gz", 5)
        first = dump_key(small)
        write_dump(tmp_path / "d.txt.gz", 500)

        assert dump_key(small) != first


class TestResuming:
    def test_a_second_pass_yields_different_candidates(self, tmp_path: Path) -> None:
        dump = write_dump(tmp_path / "dump.txt.gz", 30)
        manifest = tmp_path / "candidates.jsonl"

        first_count, first = build_manifest(dump, manifest, max_candidates=10)
        first_titles = {json.loads(line)["title"] for line in manifest.read_text().splitlines()}

        second_count, _ = build_manifest(
            dump, manifest, max_candidates=10, start_line=first.lines_read
        )
        second_titles = {json.loads(line)["title"] for line in manifest.read_text().splitlines()}

        assert first_count == second_count == 10
        assert first_titles.isdisjoint(second_titles)

    def test_reading_to_the_end_reports_exhausted(self, tmp_path: Path) -> None:
        dump = write_dump(tmp_path / "dump.txt.gz", 5)

        _, outcome = build_manifest(dump, tmp_path / "c.jsonl", max_candidates=100)

        assert outcome.exhausted

    def test_stopping_at_the_cap_does_not_report_exhausted(self, tmp_path: Path) -> None:
        # Otherwise a capped run would look like the end of the file and the
        # pipeline would stop looking at a dump it had barely read.
        dump = write_dump(tmp_path / "dump.txt.gz", 50)

        _, outcome = build_manifest(dump, tmp_path / "c.jsonl", max_candidates=10)

        assert not outcome.exhausted

    def test_resuming_past_the_end_yields_nothing(self, tmp_path: Path) -> None:
        dump = write_dump(tmp_path / "dump.txt.gz", 5)

        written, outcome = build_manifest(
            dump, tmp_path / "c.jsonl", max_candidates=10, start_line=1000
        )

        assert written == 0
        assert outcome.exhausted

    def test_the_whole_dump_is_covered_across_runs(self, tmp_path: Path) -> None:
        """Every record reached exactly once, which is the point."""
        dump = write_dump(tmp_path / "dump.txt.gz", 25)
        manifest = tmp_path / "c.jsonl"

        seen: list[str] = []
        offset = 0
        for _ in range(10):
            written, outcome = build_manifest(dump, manifest, max_candidates=7, start_line=offset)
            seen += [json.loads(line)["title"] for line in manifest.read_text().splitlines()]
            offset = outcome.lines_read
            if outcome.exhausted:
                break

        assert len(seen) == 25
        assert len(set(seen)) == 25
