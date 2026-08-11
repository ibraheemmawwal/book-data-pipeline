"""Source extractors.

Each source implements the same ``Extractor`` protocol and yields a union of
validated ``RawBook`` records and ``Rejected`` items, so one malformed record
costs that record rather than its page.
"""

from __future__ import annotations

from collections.abc import Callable

from pipeline.config import Settings
from pipeline.extract.base import (
    ExtractedItem,
    ExtractionRequest,
    Extractor,
    Rejected,
    SourceUnavailableError,
)
from pipeline.extract.googlebooks import GoogleBooksExtractor, MissingCredentialError
from pipeline.extract.gutendex import GutendexExtractor
from pipeline.extract.openlibrary import OpenLibraryExtractor
from pipeline.models.domain import SourceName

__all__ = [
    "ExtractedItem",
    "ExtractionRequest",
    "Extractor",
    "GoogleBooksExtractor",
    "GutendexExtractor",
    "MissingCredentialError",
    "OpenLibraryExtractor",
    "Rejected",
    "SourceUnavailableError",
    "build_extractor",
]

# Typed as a factory rather than a class map: the three constructors take
# different optional keywords, and only the shared (Settings) -> Extractor
# shape matters at the call site.
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
