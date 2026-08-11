"""Event envelope behaviour.

Every Kafka event carries an explicit schema_version. An unknown version is
routed to the DLQ rather than guessed, so decoding must fail loudly and
distinguishably.
"""

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from pipeline.models.domain import SourceName
from pipeline.models.events import (
    SCHEMA_VERSION,
    BookEvent,
    EventType,
    PartitionMarker,
    UnsupportedSchemaVersionError,
    decode_event,
)

RUN_ID = UUID("11111111-1111-4111-8111-111111111111")


def event_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "run_id": RUN_ID,
        "source": SourceName.GUTENDEX,
        "source_id": "1342",
        "event_type": EventType.BOOK_RAW,
        "payload": {"title": "Pride and Prejudice"},
    }
    return base | overrides


class TestBookEvent:
    def test_defaults_are_populated(self) -> None:
        event = BookEvent(**event_kwargs())  # type: ignore[arg-type]

        assert event.schema_version == SCHEMA_VERSION
        assert isinstance(event.event_id, UUID)
        assert event.emitted_at.tzinfo is UTC

    def test_event_ids_are_unique_per_event(self) -> None:
        first = BookEvent(**event_kwargs())  # type: ignore[arg-type]
        second = BookEvent(**event_kwargs())  # type: ignore[arg-type]

        assert first.event_id != second.event_id

    def test_partition_key_is_source_scoped_for_raw_events(self) -> None:
        event = BookEvent(**event_kwargs())  # type: ignore[arg-type]

        assert event.partition_key() == "gutendex:1342"

    def test_partition_key_is_identity_scoped_for_clean_events(self) -> None:
        event = BookEvent(
            **event_kwargs(
                event_type=EventType.BOOK_CLEAN,
                identity_key="isbn:9780553380163",
            )  # type: ignore[arg-type]
        )

        assert event.partition_key() == "isbn:9780553380163"

    def test_clean_event_requires_an_identity_key(self) -> None:
        with pytest.raises(ValidationError, match="identity_key"):
            BookEvent(**event_kwargs(event_type=EventType.BOOK_CLEAN))  # type: ignore[arg-type]

    def test_naive_emitted_at_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BookEvent(**event_kwargs(emitted_at=datetime(2026, 8, 20, 10, 0)))  # type: ignore[arg-type] # noqa: DTZ001

    def test_round_trips_through_json(self) -> None:
        event = BookEvent(**event_kwargs())  # type: ignore[arg-type]

        assert decode_event(event.to_json()) == event

    def test_is_immutable(self) -> None:
        event = BookEvent(**event_kwargs())  # type: ignore[arg-type]

        with pytest.raises(ValidationError):
            event.source_id = "9999"  # type: ignore[misc]


class TestPartitionMarker:
    def test_carries_only_boundary_metadata(self) -> None:
        marker = PartitionMarker(run_id=RUN_ID, topic="books.raw", partition=2, partition_count=3)

        assert marker.event_type is EventType.RUN_PARTITION_COMPLETE
        assert marker.schema_version == SCHEMA_VERSION
        # A marker is deliberately not a record-count reconciliation message:
        # extractor retries and DLQ routing make counts differ by design.
        assert not hasattr(marker, "record_count")
        assert not hasattr(marker, "expected_count")

    def test_negative_partition_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="partition"):
            PartitionMarker(run_id=RUN_ID, topic="books.raw", partition=-1, partition_count=3)

    def test_round_trips_through_json(self) -> None:
        marker = PartitionMarker(run_id=RUN_ID, topic="books.clean", partition=0, partition_count=3)

        assert decode_event(marker.to_json()) == marker


class TestDecodeEvent:
    def test_dispatches_on_event_type(self) -> None:
        raw = BookEvent(**event_kwargs())  # type: ignore[arg-type]
        marker = PartitionMarker(run_id=RUN_ID, topic="books.raw", partition=1, partition_count=3)

        assert isinstance(decode_event(raw.to_json()), BookEvent)
        assert isinstance(decode_event(marker.to_json()), PartitionMarker)

    @pytest.mark.parametrize("version", [0, 2, 99])
    def test_unknown_schema_version_raises_a_routable_error(self, version: int) -> None:
        payload = BookEvent(**event_kwargs()).model_dump(mode="json")  # type: ignore[arg-type]
        payload["schema_version"] = version

        with pytest.raises(UnsupportedSchemaVersionError) as excinfo:
            decode_event(json.dumps(payload).encode())

        # The DLQ record needs the offending version for triage.
        assert excinfo.value.schema_version == version

    def test_unsupported_version_is_distinguishable_from_malformed_input(self) -> None:
        # Malformed input is a validation rejection; an unknown version is a
        # forward-compatibility signal. They take different DLQ paths.
        assert not issubclass(UnsupportedSchemaVersionError, ValidationError)

    @pytest.mark.parametrize(
        "bad",
        [b"", b"not json", b"[]", b'{"schema_version": 1}'],
    )
    def test_malformed_input_is_rejected(self, bad: bytes) -> None:
        with pytest.raises((ValidationError, ValueError)):
            decode_event(bad)

    def test_unknown_event_type_is_rejected(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "event_id": str(uuid4()),
            "run_id": str(RUN_ID),
            "event_type": "book.invented",
            "emitted_at": "2026-08-20T10:00:00Z",
        }

        with pytest.raises(ValueError, match="event_type"):
            decode_event(json.dumps(payload).encode())


class TestPartitionMarkerTopology:
    """A marker carries the partition count it was written against.

    Completion is decided by comparing durably recorded markers against this
    number. Re-reading broker metadata instead would let a topic resized
    between runs either stall a run forever or finalise it early, so the count
    travels with the marker.
    """

    def test_partition_count_is_required(self) -> None:
        with pytest.raises(ValidationError, match="partition_count"):
            PartitionMarker(run_id=RUN_ID, topic="books.raw", partition=0)  # type: ignore[call-arg]

    def test_carries_the_topology_it_was_written_against(self) -> None:
        marker = PartitionMarker(run_id=RUN_ID, topic="books.raw", partition=2, partition_count=3)

        assert marker.partition == 2
        assert marker.partition_count == 3

    @pytest.mark.parametrize("count", [0, -1])
    def test_non_positive_partition_count_is_rejected(self, count: int) -> None:
        with pytest.raises(ValidationError, match="partition_count"):
            PartitionMarker(run_id=RUN_ID, topic="books.raw", partition=0, partition_count=count)

    @pytest.mark.parametrize(("partition", "count"), [(3, 3), (5, 3)])
    def test_partition_outside_the_declared_topology_is_rejected(
        self, partition: int, count: int
    ) -> None:
        # Partition 3 of a three-partition topic does not exist. Accepting it
        # would record an observation that can never be completed.
        with pytest.raises(ValidationError, match="partition"):
            PartitionMarker(
                run_id=RUN_ID, topic="books.raw", partition=partition, partition_count=count
            )

    def test_single_partition_topic_is_valid(self) -> None:
        marker = PartitionMarker(run_id=RUN_ID, topic="books.dlq", partition=0, partition_count=1)

        assert marker.partition_count == 1

    def test_partition_count_survives_a_json_round_trip(self) -> None:
        marker = PartitionMarker(run_id=RUN_ID, topic="books.clean", partition=1, partition_count=3)

        assert decode_event(marker.to_json()) == marker

    def test_is_still_not_a_record_count_message(self) -> None:
        marker = PartitionMarker(run_id=RUN_ID, topic="books.raw", partition=0, partition_count=3)

        assert not hasattr(marker, "record_count")
        assert not hasattr(marker, "expected_count")
