"""Pure transform layer.

No network, no filesystem, no database. Every function here is a total function
of its arguments, which is what makes the bulk of the test suite fast and makes
identity reproducible across runs and processes.
"""

from __future__ import annotations

from pipeline.transform.canonicalise import canonicalise, merge_candidates
from pipeline.transform.identity import (
    content_hash,
    fallback_identity_key,
    identity_key,
    payload_hash,
)
from pipeline.transform.isbn import is_valid_isbn13, select_canonical_isbn, to_isbn13
from pipeline.transform.normalise import (
    normalise_author,
    normalise_language,
    normalise_subject,
    normalise_title,
    parse_author_year,
    parse_year,
    select_language,
)

__all__ = [
    "canonicalise",
    "content_hash",
    "fallback_identity_key",
    "identity_key",
    "is_valid_isbn13",
    "merge_candidates",
    "normalise_author",
    "normalise_language",
    "normalise_subject",
    "normalise_title",
    "parse_author_year",
    "parse_year",
    "payload_hash",
    "select_canonical_isbn",
    "select_language",
    "to_isbn13",
]
