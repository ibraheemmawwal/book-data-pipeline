"""Domain models.

These are the validation boundary. Source APIs return inconsistent, sparse and
occasionally wrong data, so anything that reaches the transform or load stages
has already been checked here.

``RawBook`` is what one source said about one book, kept verbatim enough to
reconstruct provenance. ``CleanBook`` is the normalised, validated form whose
constraints mirror the catalogue schema's ``CHECK`` constraints exactly — a
record that would violate the database is rejected before it gets there.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ISBN13_PATTERN = r"^[0-9]{13}$"
LANGUAGE_PATTERN = r"^[a-z]{3}$"

# Canonical identity is either an ISBN-13 or a deterministic fallback digest.
# Anything else means the identity rules were bypassed.
IDENTITY_KEY_PATTERN = r"^(isbn:[0-9]{13}|fallback:[0-9a-f]{64})$"

# Gutendex dates classical authors before the common era — Homer is
# birth_year=-750 — so any non-negative lower bound would reject real records
# from the primary source. The window is wide enough for antiquity and tight
# enough to catch a parse error.
AUTHOR_YEAR_MIN = -3000
AUTHOR_YEAR_MAX = 2100

_ISBN_IDENTITY_PREFIX = "isbn:"
_FALLBACK_IDENTITY_PREFIX = "fallback:"


class SourceName(StrEnum):
    """The providers this pipeline is allowed to ingest from.

    Ordered by canonical-field priority: where two sources disagree and both
    look equally complete, the earlier one wins.
    """

    OPENLIBRARY = "openlibrary"
    GOOGLEBOOKS = "googlebooks"
    GUTENDEX = "gutendex"


NonBlankStr = Annotated[str, Field(min_length=1)]


def _require_aware_utc(value: datetime | None) -> datetime | None:
    """Reject naive datetimes and normalise everything else to UTC.

    A naive timestamp from a provider is ambiguous, and comparing it to a
    ``TIMESTAMPTZ`` column silently assumes a timezone we were never told.
    """
    if value is None:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        msg = "timestamp must be timezone-aware"
        raise ValueError(msg)
    return value.astimezone(UTC)


class _Frozen(BaseModel):
    """Immutable, strict-boundary base.

    ``extra="forbid"`` is deliberate: a field a mapper thought it was setting
    but spelled wrong is a silently dropped value, which is the hardest class
    of data bug to notice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class RawAuthor(_Frozen):
    """An author as one source described them.

    ``source_author_id`` is kept because ``author_sources`` needs it; author
    identity across providers cannot be recovered from a display name alone.
    """

    name: NonBlankStr
    source_author_id: str | None = None

    # Gutendex publishes these for nearly every author. They are the catalogue's
    # densest calendar dimension, because Gutendex carries no publication year.
    birth_year: Annotated[int, Field(ge=AUTHOR_YEAR_MIN, le=AUTHOR_YEAR_MAX)] | None = None
    death_year: Annotated[int, Field(ge=AUTHOR_YEAR_MIN, le=AUTHOR_YEAR_MAX)] | None = None

    @model_validator(mode="after")
    def _check_lifespan_order(self) -> Self:
        """A death before a birth is a parse error, not a biography."""
        if (
            self.birth_year is not None
            and self.death_year is not None
            and self.death_year < self.birth_year
        ):
            msg = f"death_year {self.death_year} precedes birth_year {self.birth_year}"
            raise ValueError(msg)
        return self


class RawBook(_Frozen):
    """One source's account of one book, before normalisation.

    Fields are permissive by design — ``published`` is whatever string the
    provider sent, ``isbns`` are unvalidated. Tightening happens in transform,
    so a malformed value becomes a recorded rejection rather than an import
    that dies halfway through a page.
    """

    source: SourceName
    source_id: NonBlankStr

    title: NonBlankStr
    subtitle: str | None = None
    authors: list[RawAuthor] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)

    isbns: list[str] = Field(default_factory=list)
    # Plural because sources are plural. Open Library reports one entry per
    # edition of a work, so collapsing the list here would silently pick an
    # arbitrary edition's language for the whole book.
    languages: list[str] = Field(default_factory=list)
    published: str | None = None
    publisher: str | None = None
    page_count: int | None = None
    description: str | None = None
    cover_url: str | None = None

    # Gutendex only. The single popularity signal any source gives us, and the
    # basis for the catalogue's ranking analytics.
    download_count: Annotated[int, Field(ge=0)] | None = None

    source_updated: datetime | None = None
    raw_payload: dict[str, Any]

    @field_validator("source_id", mode="before")
    @classmethod
    def _coerce_source_id(cls, value: object) -> object:
        """Gutendex ids are integers; Open Library's are strings."""
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value

    @field_validator("source_updated")
    @classmethod
    def _check_source_updated(cls, value: datetime | None) -> datetime | None:
        return _require_aware_utc(value)


class CleanBook(_Frozen):
    """A validated, normalised record ready for canonical resolution.

    Every constraint here mirrors one in the catalogue schema. Provenance is
    retained so the load stage can attribute the row to a ``book_sources``
    entry without a second lookup.
    """

    source: SourceName
    source_id: NonBlankStr

    identity_key: Annotated[str, Field(pattern=IDENTITY_KEY_PATTERN)]

    title: NonBlankStr
    normalised_title: NonBlankStr
    subtitle: str | None = None

    isbn13: Annotated[str, Field(pattern=ISBN13_PATTERN)] | None = None
    published_year: Annotated[int, Field(ge=1400, le=2100)] | None = None
    page_count: Annotated[int, Field(gt=0)] | None = None
    language: Annotated[str, Field(pattern=LANGUAGE_PATTERN)] | None = None

    publisher: str | None = None
    description: str | None = None
    cover_url: str | None = None
    download_count: Annotated[int, Field(ge=0)] | None = None

    authors: list[RawAuthor] = Field(default_factory=list)
    normalised_first_author: str | None = None
    subjects: list[str] = Field(default_factory=list)

    source_updated: datetime | None = None
    raw_payload: dict[str, Any]

    @field_validator("source_updated")
    @classmethod
    def _check_source_updated(cls, value: datetime | None) -> datetime | None:
        return _require_aware_utc(value)

    @model_validator(mode="after")
    def _check_identity_key_agrees_with_isbn(self) -> Self:
        """An identity key that disagrees with its own ISBN merges the wrong books.

        This is cheap to check here and expensive to debug once rows exist.
        """
        if self.identity_key.startswith(_ISBN_IDENTITY_PREFIX):
            expected = self.identity_key.removeprefix(_ISBN_IDENTITY_PREFIX)
            if self.isbn13 != expected:
                msg = f"identity_key {self.identity_key!r} does not match isbn13 {self.isbn13!r}"
                raise ValueError(msg)
        elif self.identity_key.startswith(_FALLBACK_IDENTITY_PREFIX) and self.isbn13 is not None:
            msg = (
                f"identity_key {self.identity_key!r} is a fallback key but "
                f"isbn13 {self.isbn13!r} is present; an ISBN must use an isbn: key"
            )
            raise ValueError(msg)
        return self

    def has_canonical_isbn(self) -> bool:
        """Whether this record carries the strong canonical identity."""
        return self.isbn13 is not None


def is_isbn_identity(identity_key: str) -> bool:
    """Whether ``identity_key`` is an ISBN identity rather than a fallback digest."""
    return re.fullmatch(r"isbn:[0-9]{13}", identity_key) is not None
