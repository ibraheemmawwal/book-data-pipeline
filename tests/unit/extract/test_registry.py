"""The source registry."""

from __future__ import annotations

import pytest

from pipeline.config import Settings
from pipeline.extract import build_extractor
from pipeline.extract.base import Extractor
from pipeline.models.domain import SourceName


@pytest.mark.parametrize("source", list(SourceName))
def test_every_source_has_an_extractor(source: SourceName, settings: Settings) -> None:
    # A SourceName with no extractor is a source that silently never runs.
    extractor = build_extractor(source, settings)

    assert isinstance(extractor, Extractor)
    assert extractor.source_name is source
