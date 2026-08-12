"""Fetching the Open Library dump.

The published editions dump is around 12 GB compressed and republished monthly.
A pipeline that cannot obtain its own input is not automated, so this fetches
it — but downloading 12 GB on every scheduled run to read the next few hundred
lines would be absurd, so the file is cached and only refreshed when the
published one changes.

A slice is supported for the same reason a full download is: both are honest,
and which is appropriate depends on whether you are demonstrating the pipeline
or running it.
"""

from __future__ import annotations

import gzip
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger(__name__)

DUMP_URL = "https://openlibrary.org/data/ol_dump_editions_latest.txt.gz"
CHUNK = 1 << 20
# 16 + MAX_WBITS tells zlib the stream carries a gzip header rather than a bare
# deflate one.
_GZIP_WINDOW = 16 + zlib.MAX_WBITS


@dataclass(frozen=True, slots=True)
class FetchResult:
    """What the fetch did, so a task can report it without re-statting."""

    path: Path
    bytes_on_disk: int
    downloaded: bool
    reason: str


def _remote_size(url: str, timeout: float) -> int | None:
    """The published file's size, or None when it cannot be read.

    Used as a cheap change signal: the dump is republished monthly and its size
    always moves, so a matching size means the cached copy is the same edition.
    A checksum would be stronger and would require downloading the file to
    compute, which is the cost this is avoiding.
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.head(url)
            response.raise_for_status()
            length = response.headers.get("content-length")
            return int(length) if length else None
    except (httpx.HTTPError, ValueError):
        return None


def fetch_dump(
    destination: Path,
    *,
    url: str = DUMP_URL,
    max_lines: int | None = None,
    timeout: float = 60.0,
) -> FetchResult:
    """Ensure a usable dump exists at ``destination``.

    Args:
        max_lines: Stop after this many records and re-compress. A byte-range
            prefix would be cheaper and unusable — it is a truncated gzip, which
            the reader refuses outright, correctly. Re-compressing whole lines
            yields genuine records in a valid file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    published = _remote_size(url, timeout)

    if destination.exists():
        local = destination.stat().st_size
        # A sliced local copy is smaller than the published file by design, so
        # size equality cannot be the test for it; its presence is enough.
        if max_lines is not None:
            return FetchResult(destination, local, False, "cached slice present")
        if published is not None and local == published:
            return FetchResult(destination, local, False, "cached copy matches published size")

    logger.info("dump.fetching", url=url, max_lines=max_lines)
    partial = destination.with_suffix(destination.suffix + ".partial")

    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
        response.raise_for_status()
        if max_lines is None:
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(CHUNK):
                    handle.write(chunk)
        else:
            # The dump is a .gz *file*, not a gzip-encoded response, so the
            # client does not decompress it and iter_lines() would hand back
            # compressed bytes. Decompress the stream here.
            written = 0
            decompressor = zlib.decompressobj(_GZIP_WINDOW)
            remainder = b""
            with gzip.open(partial, "wt", encoding="utf-8") as out:
                for chunk in response.iter_bytes(CHUNK):
                    remainder += decompressor.decompress(chunk)
                    *complete, remainder = remainder.split(b"\n")
                    for raw in complete:
                        out.write(raw.decode("utf-8", "replace") + "\n")
                        written += 1
                        if written >= max_lines:
                            break
                    if written >= max_lines:
                        break

    # Atomic: a crash mid-download must not leave a half-file that looks
    # complete to the next run.
    shutil.move(str(partial), str(destination))
    size = destination.stat().st_size
    logger.info("dump.fetched", path=str(destination), bytes=size)
    return FetchResult(destination, size, True, "downloaded")
