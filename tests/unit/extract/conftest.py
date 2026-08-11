"""Shared fixtures for extractor tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.config import Settings

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    """Read a captured API response."""
    with (FIXTURES / name).open() as handle:
        payload: dict[str, Any] = json.load(handle)
        return payload


@pytest.fixture
def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        database_url="postgresql+psycopg://u:p@localhost:5432/catalogue",
        openlibrary_contact_email="owner@example.com",
        googlebooks_api_key="test-key",
        http_connect_timeout_seconds=0.5,
        http_read_timeout_seconds=1.0,
    )
