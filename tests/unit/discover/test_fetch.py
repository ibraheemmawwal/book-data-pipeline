"""Fetching the Open Library dump.

The real file is around 12 GB, so every test here works on a small gzip built
in the test itself. What is being pinned is not the download — httpx does that
— but the four decisions around it: when to skip the download entirely, that a
slice is a *valid* gzip rather than a truncated one, and that a crash cannot
leave a half-file the next run would trust.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import httpx
import pytest
import respx

from pipeline.discover.fetch import DUMP_URL, fetch_dump


def dump_bytes(lines: int) -> bytes:
    body = "".join(
        f'/type/edition\t/books/OL{n}M\t1\t2024-01-01\t{{"title": "Book {n}"}}\n'
        for n in range(lines)
    )
    return gzip.compress(body.encode())


def lines_in(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.read().splitlines()


@respx.mock
def test_it_downloads_when_nothing_is_cached(tmp_path: Path) -> None:
    payload = dump_bytes(5)
    respx.head(DUMP_URL).mock(
        return_value=httpx.Response(200, headers={"content-length": str(len(payload))})
    )
    respx.get(DUMP_URL).mock(return_value=httpx.Response(200, content=payload))
    destination = tmp_path / "dump.txt.gz"

    result = fetch_dump(destination)

    assert result.downloaded is True
    assert destination.exists()
    assert len(lines_in(destination)) == 5


@respx.mock
def test_a_matching_size_skips_the_download(tmp_path: Path) -> None:
    """The whole reason a HEAD request happens at all.

    The dump is republished monthly and 12 GB; re-downloading it on every
    scheduled run to read a few hundred lines is the failure mode this avoids.
    """
    payload = dump_bytes(3)
    destination = tmp_path / "dump.txt.gz"
    destination.write_bytes(payload)
    respx.head(DUMP_URL).mock(
        return_value=httpx.Response(200, headers={"content-length": str(len(payload))})
    )
    route = respx.get(DUMP_URL).mock(return_value=httpx.Response(200, content=payload))

    result = fetch_dump(destination)

    assert result.downloaded is False
    assert "matches published size" in result.reason
    assert route.call_count == 0


@respx.mock
def test_a_different_size_downloads_again(tmp_path: Path) -> None:
    fresh = dump_bytes(9)
    destination = tmp_path / "dump.txt.gz"
    destination.write_bytes(dump_bytes(2))
    respx.head(DUMP_URL).mock(
        return_value=httpx.Response(200, headers={"content-length": str(len(fresh))})
    )
    respx.get(DUMP_URL).mock(return_value=httpx.Response(200, content=fresh))

    result = fetch_dump(destination)

    assert result.downloaded is True
    assert len(lines_in(destination)) == 9


@respx.mock
def test_an_unreadable_head_still_downloads(tmp_path: Path) -> None:
    # Not knowing the published size is a reason to fetch, never a reason to
    # trust whatever is on disk.
    payload = dump_bytes(4)
    respx.head(DUMP_URL).mock(side_effect=httpx.ConnectError("no route"))
    respx.get(DUMP_URL).mock(return_value=httpx.Response(200, content=payload))
    destination = tmp_path / "dump.txt.gz"
    destination.write_bytes(dump_bytes(1))

    result = fetch_dump(destination)

    assert result.downloaded is True
    assert len(lines_in(destination)) == 4


@respx.mock
def test_a_missing_content_length_still_downloads(tmp_path: Path) -> None:
    payload = dump_bytes(2)
    respx.head(DUMP_URL).mock(return_value=httpx.Response(200))
    respx.get(DUMP_URL).mock(return_value=httpx.Response(200, content=payload))
    destination = tmp_path / "dump.txt.gz"

    assert fetch_dump(destination).downloaded is True


class TestSlicing:
    @respx.mock
    def test_it_stops_at_the_line_limit(self, tmp_path: Path) -> None:
        respx.head(DUMP_URL).mock(return_value=httpx.Response(200))
        respx.get(DUMP_URL).mock(return_value=httpx.Response(200, content=dump_bytes(500)))
        destination = tmp_path / "dump.txt.gz"

        fetch_dump(destination, max_lines=10)

        assert len(lines_in(destination)) == 10

    @respx.mock
    def test_the_slice_is_a_valid_gzip_of_whole_records(self, tmp_path: Path) -> None:
        """Why the slice is re-compressed rather than byte-ranged.

        A byte-range prefix of a .gz is a truncated gzip, which the reader
        refuses outright — correctly. Re-compressing whole lines is what makes
        a slice genuinely readable.
        """
        respx.head(DUMP_URL).mock(return_value=httpx.Response(200))
        respx.get(DUMP_URL).mock(return_value=httpx.Response(200, content=dump_bytes(100)))
        destination = tmp_path / "dump.txt.gz"

        fetch_dump(destination, max_lines=7)

        lines = lines_in(destination)
        assert len(lines) == 7
        # Every line is a whole record, not a fragment of one.
        assert all(line.count("\t") == 4 for line in lines)
        assert lines[0].endswith('{"title": "Book 0"}')

    @respx.mock
    def test_a_present_slice_is_not_refetched(self, tmp_path: Path) -> None:
        # A slice is smaller than the published file by design, so size
        # equality cannot be the test for it; presence has to be.
        destination = tmp_path / "dump.txt.gz"
        destination.write_bytes(dump_bytes(3))
        respx.head(DUMP_URL).mock(return_value=httpx.Response(200))
        route = respx.get(DUMP_URL).mock(return_value=httpx.Response(200, content=dump_bytes(50)))

        result = fetch_dump(destination, max_lines=10)

        assert result.downloaded is False
        assert "cached slice" in result.reason
        assert route.call_count == 0

    @respx.mock
    def test_a_short_source_yields_what_there_is(self, tmp_path: Path) -> None:
        respx.head(DUMP_URL).mock(return_value=httpx.Response(200))
        respx.get(DUMP_URL).mock(return_value=httpx.Response(200, content=dump_bytes(3)))
        destination = tmp_path / "dump.txt.gz"

        fetch_dump(destination, max_lines=100)

        assert len(lines_in(destination)) == 3


class TestFailure:
    @respx.mock
    def test_an_http_error_raises(self, tmp_path: Path) -> None:
        respx.head(DUMP_URL).mock(return_value=httpx.Response(200))
        respx.get(DUMP_URL).mock(return_value=httpx.Response(503))

        with pytest.raises(httpx.HTTPStatusError):
            fetch_dump(tmp_path / "dump.txt.gz")

    @respx.mock
    def test_a_failed_download_leaves_no_usable_file(self, tmp_path: Path) -> None:
        """The reason the download lands on .partial and is then moved.

        A half-written file at the real path looks complete to the next run,
        which then reads a truncated dump and reports success.
        """
        respx.head(DUMP_URL).mock(return_value=httpx.Response(200))
        respx.get(DUMP_URL).mock(return_value=httpx.Response(500))
        destination = tmp_path / "dump.txt.gz"

        with pytest.raises(httpx.HTTPStatusError):
            fetch_dump(destination)

        assert not destination.exists()

    @respx.mock
    def test_it_creates_the_parent_directory(self, tmp_path: Path) -> None:
        respx.head(DUMP_URL).mock(return_value=httpx.Response(200))
        respx.get(DUMP_URL).mock(return_value=httpx.Response(200, content=dump_bytes(1)))
        destination = tmp_path / "nested" / "deeper" / "dump.txt.gz"

        assert fetch_dump(destination).path.exists()
