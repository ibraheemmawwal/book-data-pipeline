"""The source registry."""

from __future__ import annotations

import pytest

from pipeline.config import Settings
from pipeline.extract import BULK_SOURCES, build_extractor
from pipeline.extract.base import Extractor
from pipeline.models.domain import SourceName


@pytest.mark.parametrize("source", BULK_SOURCES)
def test_every_bulk_source_has_an_extractor(source: SourceName, settings: Settings) -> None:
    # A bulk source with no extractor is a source that silently never runs.
    extractor = build_extractor(source, settings)

    assert isinstance(extractor, Extractor)
    assert extractor.source_name is source


def test_goodreads_is_not_a_bulk_source() -> None:
    # It has no supported bulk enumeration contract, so it resolves one
    # candidate at a time and is reached through the resolver instead.
    assert SourceName.GOODREADS not in BULK_SOURCES


def test_bulk_sources_covers_everything_except_goodreads() -> None:
    # Guards against a new SourceName being added and silently never running.
    assert set(BULK_SOURCES) == set(SourceName) - {SourceName.GOODREADS}
