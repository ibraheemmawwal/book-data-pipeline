"""Candidate discovery from an Open Library data dump.

Open Library's own guidance points at dumps rather than the search API for bulk
access, so this is where the catalogue's breadth comes from. The dump is tens
of gigabytes compressed, which shapes everything here: it is read as a stream,
one line at a time, and never held in memory.

The other constraint is determinism. For a pinned checksum the same dump must
always produce the same manifest — otherwise a rerun resolves a different set
of books and nothing downstream is reproducible. That is why the checksum is
verified before a manifest is published, and why the manifest is written to a
temporary file and renamed only on success: a failed run must not leave a
half-trusted manifest for the next one to pick up.

Rows are tab-separated with five columns — type, key, revision, last modified,
and a JSON document. A malformed row costs that row and nothing else; one
corrupt line in forty million is expected, and aborting the stream over it
would throw away the rest of the dump.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import structlog

from pipeline.models.domain import CandidateBook
from pipeline.transform.isbn import to_isbn13

logger = structlog.get_logger(__name__)

EDITION_TYPE = "/type/edition"
DUMP_COLUMNS = 5
READ_CHUNK_BYTES = 1024 * 1024

# "by Frank Herbert" / "par Quelqu'un" — the leading preposition is noise once
# the name is used as a search term.
_BY_PREFIXES = ("by ", "par ", "von ", "de ")


class TruncatedDumpError(Exception):
    """The dump ended mid-stream.

    Almost always an interrupted download. Continuing would silently discover
    from a partial file, which breaks the determinism the pinned checksum is
    supposed to guarantee — so this fails loudly rather than quietly finding
    fewer books.
    """


class ChecksumMismatchError(Exception):
    """The dump on disk is not the one that was pinned.

    Discovery is only reproducible against a known input; silently accepting a
    different dump would make every downstream guarantee about determinism
    false without anything failing.
    """


def verify_checksum(path: Path, expected_sha256: str) -> None:
    """Confirm a dump matches its pinned digest.

    Raises:
        ChecksumMismatchError: the file does not match.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            digest.update(chunk)

    actual = digest.hexdigest()
    if actual.lower() != expected_sha256.strip().lower():
        msg = (
            f"{path.name} does not match the pinned checksum: "
            f"expected {expected_sha256.strip().lower()}, got {actual}"
        )
        raise ChecksumMismatchError(msg)


def _author_names(document: dict[str, Any]) -> list[str]:
    """Recover resolvable author names.

    Edition records carry author *keys*, not names, and a key cannot be turned
    into a search query — the TRD is explicit that a key alone is not a
    resolvable candidate. ``by_statement`` is the only place in an edition
    record a usable name appears.
    """
    statement = document.get("by_statement")
    if not isinstance(statement, str):
        return []

    name = statement.strip().rstrip(".").strip()
    lowered = name.lower()
    for prefix in _BY_PREFIXES:
        if lowered.startswith(prefix):
            name = name[len(prefix) :].strip()
            break
    return [name] if name else []


def _isbns(document: dict[str, Any]) -> list[str]:
    """Every ISBN in the record, validated.

    An ISBN that fails its checksum is a typo, not lookup material, so it is
    dropped here rather than sent to a source that will not find it.
    """
    raw: list[str] = []
    for field in ("isbn_13", "isbn_10"):
        values = document.get(field)
        if isinstance(values, list):
            raw.extend(value for value in values if isinstance(value, str))

    seen: dict[str, None] = {}
    for value in raw:
        converted = to_isbn13(value)
        if converted is not None:
            seen[converted] = None
    return list(seen)


def _languages(document: dict[str, Any]) -> list[str]:
    """ISO codes from ``[{"key": "/languages/eng"}]``."""
    values = document.get("languages")
    if not isinstance(values, list):
        return []
    codes = []
    for entry in values:
        if isinstance(entry, dict) and isinstance(entry.get("key"), str):
            codes.append(entry["key"].rsplit("/", 1)[-1])
    return codes


def _first_key(document: dict[str, Any], field: str) -> str | None:
    values = document.get(field)
    if isinstance(values, list) and values:
        first = values[0]
        if isinstance(first, dict):
            key = first.get("key")
            if isinstance(key, str):
                return key
    return None


def _to_candidate(document: dict[str, Any]) -> CandidateBook | None:
    """Build a candidate, or ``None`` if the record cannot be looked up.

    The bar is a title plus something to search *with*: either a usable author
    name or a valid ISBN. A title on its own matches thousands of books, and a
    record we cannot resolve is a request we should never send.
    """
    title = document.get("title")
    if not isinstance(title, str) or not title.strip():
        return None

    key = document.get("key")
    if not isinstance(key, str) or not key:
        return None

    authors = _author_names(document)
    isbns = _isbns(document)
    if not authors and not isbns:
        return None

    return CandidateBook(
        candidate_key=key,
        title=title.strip(),
        authors=authors,
        isbns=isbns,
        openlibrary_edition_key=key,
        openlibrary_work_key=_first_key(document, "works"),
        languages=_languages(document),
        discovery_payload=document,
    )


def _lines(handle: Any, path: Path) -> Iterator[str]:
    """Iterate lines, turning gzip corruption into a named failure.

    An interrupted download surfaces as a bare EOFError from deep inside the
    decompressor, which says nothing useful about what went wrong.
    """
    try:
        yield from handle
    except (EOFError, gzip.BadGzipFile, OSError) as error:
        msg = f"{path.name} ended mid-stream or is not valid gzip: {error}"
        raise TruncatedDumpError(msg) from error


def stream_candidates(
    path: Path,
    *,
    languages: frozenset[str] | None = None,
    max_candidates: int | None = None,
) -> Iterator[CandidateBook]:
    """Stream resolvable candidates out of a gzipped dump.

    A generator by design: the real file is tens of gigabytes and materialising
    it would defeat the point.

    Records with no language are kept even when filtering. Most of the dump has
    no language field, and dropping those would discard the majority of the
    catalogue to enforce a constraint the record never stated.
    """
    emitted = 0
    skipped = 0

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in _lines(handle, path):
            if max_candidates is not None and emitted >= max_candidates:
                break

            columns = line.rstrip("\n").split("\t")
            if len(columns) != DUMP_COLUMNS or columns[0] != EDITION_TYPE:
                skipped += 1
                continue

            try:
                document = json.loads(columns[4])
            except json.JSONDecodeError:
                # One corrupt line must not cost the other forty million.
                skipped += 1
                continue

            if not isinstance(document, dict):
                skipped += 1
                continue

            candidate = _to_candidate(document)
            if candidate is None:
                skipped += 1
                continue

            if languages is not None:
                found = candidate.languages
                if found and not (set(found) & languages):
                    skipped += 1
                    continue

            emitted += 1
            yield candidate

    logger.info("discovery.stream_complete", emitted=emitted, skipped=skipped)


def build_manifest(
    path: Path,
    manifest_path: Path,
    *,
    languages: frozenset[str] | None = None,
    max_candidates: int | None = None,
    expected_sha256: str | None = None,
) -> int:
    """Materialise a deterministic candidate manifest.

    Written to a temporary file beside the target and renamed only after the
    checksum verifies, so a failed or interrupted run leaves nothing behind for
    the next one to mistake for a good manifest. ``os.replace`` is atomic on
    the same filesystem, which is why the temporary file is a sibling.

    Returns the number of candidates written.

    Raises:
        ChecksumMismatchError: ``expected_sha256`` was given and did not match.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".partial")

    written = 0
    try:
        with temporary.open("w", encoding="utf-8") as out:
            for candidate in stream_candidates(
                path, languages=languages, max_candidates=max_candidates
            ):
                # sort_keys so the same dump always yields byte-identical
                # output; a manifest that shuffled would break reproducibility
                # without changing a single candidate.
                out.write(
                    json.dumps(
                        candidate.model_dump(mode="json"),
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1

        if expected_sha256 is not None:
            verify_checksum(path, expected_sha256)

        temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)

    logger.info("discovery.manifest_written", path=str(manifest_path), candidates=written)
    return written


def read_manifest(manifest_path: Path) -> Iterator[CandidateBook]:
    """Stream candidates back out of a manifest."""
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield CandidateBook.model_validate_json(line)
