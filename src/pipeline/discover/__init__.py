"""Candidate discovery.

Discovery answers "which books are worth resolving", which is a different and
weaker claim than "a source observed these fields". It produces
``CandidateBook`` values, never canonical books.
"""

from __future__ import annotations

from pipeline.discover.openlibrary_dump import (
    ChecksumMismatchError,
    TruncatedDumpError,
    build_manifest,
    read_manifest,
    stream_candidates,
    verify_checksum,
)

__all__ = [
    "ChecksumMismatchError",
    "TruncatedDumpError",
    "build_manifest",
    "read_manifest",
    "stream_candidates",
    "verify_checksum",
]
