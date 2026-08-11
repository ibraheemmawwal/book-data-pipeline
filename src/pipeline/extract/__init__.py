"""Source extractors.

Each source implements the same ``Extractor`` protocol and yields a union of
validated ``RawBook`` records and ``Rejected`` items, so one malformed record
costs that record rather than its page.
"""

from __future__ import annotations

from collections.abc import Callable

from pipeline.config import Settings
from pipeline.extract import goodreads, googlebooks, gutendex, openlibrary
from pipeline.extract.base import (
    ExtractedItem,
    ExtractionRequest,
    Extractor,
    Rejected,
    SourceUnavailableError,
)
from pipeline.extract.goodreads import GoodreadsExtractor, GoodreadsNotAcceptedError
from pipeline.extract.googlebooks import GoogleBooksExtractor, MissingCredentialError
from pipeline.extract.gutendex import GutendexExtractor
from pipeline.extract.openlibrary import OpenLibraryExtractor
from pipeline.models.domain import SourceName

__all__ = [
    "BULK_SOURCES",
    "ExtractedItem",
    "ExtractionRequest",
    "Extractor",
    "GoodreadsExtractor",
    "GoodreadsNotAcceptedError",
    "GoogleBooksExtractor",
    "GutendexExtractor",
    "MissingCredentialError",
    "OpenLibraryExtractor",
    "Rejected",
    "SourceUnavailableError",
    "build_extractor",
    "map_payload",
]

# Typed as a factory rather than a class map: the three constructors take
# different optional keywords, and only the shared (Settings) -> Extractor
# shape matters at the call site.
# Goodreads is deliberately absent. It has no supported bulk enumeration
# contract, so it resolves one candidate at a time and is reached through the
# resolver rather than by fetching pages of results.
BULK_SOURCES = (SourceName.OPENLIBRARY, SourceName.GOOGLEBOOKS, SourceName.GUTENDEX)

_EXTRACTORS: dict[SourceName, Callable[[Settings], Extractor]] = {
    SourceName.GUTENDEX: GutendexExtractor,
    SourceName.OPENLIBRARY: OpenLibraryExtractor,
    SourceName.GOOGLEBOOKS: GoogleBooksExtractor,
}


def build_extractor(source: SourceName, settings: Settings) -> Extractor:
    """Construct the extractor for ``source``.

    Keeps the DAG and CLI from importing concrete classes, so adding a source
    is one entry here rather than a change at every call site.
    """
    return _EXTRACTORS[source](settings)


# Mapping a stored payload back to a RawBook needs no Settings and no HTTP
# client. The load layer relies on this: canonical fields are recomputed from
# every book_sources row attached to a book, and those rows hold raw payloads
# from runs that finished long ago.
_MAPPERS: dict[SourceName, Callable[[object], ExtractedItem]] = {
    SourceName.GOODREADS: goodreads.map_payload,
    SourceName.GUTENDEX: gutendex.map_payload,
    SourceName.OPENLIBRARY: openlibrary.map_payload,
    SourceName.GOOGLEBOOKS: googlebooks.map_payload,
}


def map_payload(source: SourceName, payload: object) -> ExtractedItem:
    """Re-map a stored raw payload using the source's own mapper."""
    return _MAPPERS[source](payload)
