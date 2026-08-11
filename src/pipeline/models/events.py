"""Kafka event envelope.

Every event carries an explicit ``schema_version``. An unknown version is
routed to the DLQ rather than guessed, which is why decoding distinguishes
three outcomes: a valid event, a malformed payload, and a payload from a
future producer. Those take different DLQ paths and need different triage.

Partition markers are pure boundary signals. They deliberately carry no record
or topology count. Consumers compare durable observations with the expectation
frozen in ``run_topic_partitions`` before marker emission; an event is never an
authority for broker topology.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipeline.models.domain import IDENTITY_KEY_PATTERN, SourceName

SCHEMA_VERSION = 1


class EventType(StrEnum):
    BOOK_RAW = "book.raw"
    BOOK_CLEAN = "book.clean"
    RUN_PARTITION_COMPLETE = "run.partition_complete"


class UnsupportedSchemaVersionError(Exception):
    """A producer emitted a schema version this consumer does not understand.

    Intentionally not a ``ValidationError``: the payload may be perfectly
    well-formed for a newer contract. Guessing at its meaning is worse than
    routing it to the DLQ, and the consumer must be able to tell the two cases
    apart to report the right rejection code.
    """

    def __init__(self, schema_version: object) -> None:
        self.schema_version = schema_version
        super().__init__(
            f"unsupported schema_version {schema_version!r}; this consumer supports "
            f"{SCHEMA_VERSION}"
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _Envelope(BaseModel):
    """Fields common to every event on every topic."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, Field(strict=True)] = SCHEMA_VERSION
    event_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    event_type: EventType
    emitted_at: datetime = Field(default_factory=_utc_now)

    @field_validator("schema_version")
    @classmethod
    def _supported_schema_version(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(value)
        return value

    @field_validator("emitted_at")
    @classmethod
    def _require_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            msg = "emitted_at must be timezone-aware"
            raise ValueError(msg)
        return value.astimezone(UTC)

    def to_json(self) -> bytes:
        """Serialise for the wire."""
        return self.model_dump_json().encode()


class BookEvent(_Envelope):
    """A book record in transit, raw or clean."""

    event_type: EventType = EventType.BOOK_RAW
    source: SourceName
    source_id: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)

    # Present on clean events only; assigned once canonical identity is known.
    identity_key: Annotated[str, Field(pattern=IDENTITY_KEY_PATTERN)] | None = None

    @model_validator(mode="after")
    def _check_event_type_and_identity(self) -> Self:
        if self.event_type is EventType.RUN_PARTITION_COMPLETE:
            msg = "a BookEvent cannot use the run.partition_complete event_type"
            raise ValueError(msg)
        if self.event_type is EventType.BOOK_CLEAN and self.identity_key is None:
            msg = "identity_key is required on book.clean events"
            raise ValueError(msg)
        return self

    def partition_key(self) -> str:
        """The Kafka message key.

        Raw events key on ingestion identity, which never changes. Clean events
        key on canonical identity, which *can* change when a fallback book is
        promoted to an ISBN — so this distributes work but is not an immutable
        ordering key. Nothing in this release depends on cross-run per-book
        ordering; the transactional load algorithm is authoritative.
        """
        if self.event_type is EventType.BOOK_CLEAN:
            # Guaranteed non-None by _check_event_type_and_identity.
            assert self.identity_key is not None
            return self.identity_key
        return f"{self.source.value}:{self.source_id}"


class PartitionMarker(_Envelope):
    """A run boundary signal for one partition of one topic.

    Seeing a marker on every partition proves the consumer has observed every
    earlier event for that run, including duplicates from task retries.
    """

    event_type: EventType = EventType.RUN_PARTITION_COMPLETE
    topic: str = Field(min_length=1)
    partition: int = Field(ge=0)

    @field_validator("event_type")
    @classmethod
    def _must_be_a_marker(cls, value: EventType) -> EventType:
        if value is not EventType.RUN_PARTITION_COMPLETE:
            msg = "a PartitionMarker must use the run.partition_complete event_type"
            raise ValueError(msg)
        return value

    @field_validator("topic")
    @classmethod
    def _topic_has_a_completion_boundary(cls, value: str) -> str:
        if value not in {"books.raw", "books.clean"}:
            msg = "partition markers are valid only for books.raw or books.clean"
            raise ValueError(msg)
        return value


def decode_event(data: bytes) -> BookEvent | PartitionMarker:
    """Decode a wire payload into the model its ``event_type`` names.

    Raises:
        UnsupportedSchemaVersionError: the version is not understood.
        ValueError: the payload is malformed or names an unknown event type.
            ``pydantic.ValidationError`` is a ``ValueError`` subclass, so field
            level failures surface here too.
    """
    document = json.loads(data)
    if not isinstance(document, dict):
        msg = f"event payload must be a JSON object, got {type(document).__name__}"
        raise ValueError(msg)

    # Version first: a future contract may have renamed or removed anything
    # below, so no other field can be trusted until the version matches.
    version = document.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(version)

    raw_type = document.get("event_type")
    if raw_type == EventType.RUN_PARTITION_COMPLETE.value:
        return PartitionMarker.model_validate(document)
    if raw_type in {EventType.BOOK_RAW.value, EventType.BOOK_CLEAN.value}:
        return BookEvent.model_validate(document)

    msg = f"unknown event_type {raw_type!r}"
    raise ValueError(msg)
