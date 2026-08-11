"""Canonical load layer."""

from __future__ import annotations

from pipeline.load.postgres import (
    CatalogueLoader,
    LoadResult,
    record_attempts,
    record_rejection,
)

__all__ = ["CatalogueLoader", "LoadResult", "record_attempts", "record_rejection"]
