"""The Kafka adapters against a real broker.

The unit tests pin the logic with a fake client; these pin the things a fake
cannot: that a produced event actually comes back, that a key really does route
a book's updates to one partition, that a marker written to partition 2 is read
from partition 2, and that an uncommitted offset really is redelivered.

Marked `kafka` so it can be selected and skipped independently — a broker is a
much heavier dependency than PostgreSQL.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from pipeline.models.domain import SourceName
from pipeline.models.events import BookEvent, PartitionMarker

pytestmark = [pytest.mark.integration, pytest.mark.kafka]

pytest.importorskip("confluent_kafka", reason="requires the kafka client")

PARTITIONS = 3
RUN_ID = uuid.uuid4()


@pytest.fixture(scope="module")
def broker() -> Iterator[str]:
    """A throwaway KRaft broker."""
    # The container's own default image: its readiness probe and env wiring
    # are written for Confluent's build, and pointing it at apache/kafka makes
    # the broker exit before it ever reports ready. Compose runs apache/kafka
    # in KRaft mode; what is under test here is our adapter, not the packaging.
    from testcontainers.community.kafka import KafkaContainer

    with KafkaContainer() as container:
        yield container.get_bootstrap_server()


@pytest.fixture
def topic(broker: str) -> str:
    """A fresh topic with the declared partition count.

    Created explicitly, never auto-created: an auto-created topic gets one
    partition, and the marker written to partition 2 would have nowhere to go.
    """
    from confluent_kafka.admin import AdminClient, NewTopic

    name = f"books.raw.{uuid.uuid4().hex[:8]}"
    admin = AdminClient({"bootstrap.servers": broker})
    futures = admin.create_topics([NewTopic(name, num_partitions=PARTITIONS, replication_factor=1)])
    futures[name].result(timeout=30)
    return name


def book(source_id: str = "1") -> BookEvent:
    return BookEvent(
        run_id=RUN_ID,
        source=SourceName.GUTENDEX,
        source_id=source_id,
        payload={"id": source_id, "title": f"Book {source_id}"},
    )


def drain(source: Any, expected: int) -> list[Any]:
    """Consume until `expected` events arrive or the topic goes quiet.

    The source stops itself after a few idle polls, so a topic with nothing on
    it ends the loop instead of hanging — a caller can only call stop() from
    inside the iteration, which is no help when nothing is ever yielded.
    """
    seen: list[Any] = []
    for event in source.consume():
        seen.append(event)
        if len(seen) >= expected:
            source.stop()
    return seen


class TestRoundTrip:
    def test_a_produced_event_comes_back(self, broker: str, topic: str) -> None:
        from pipeline.messaging.kafka import KafkaSink, KafkaSource

        sink = KafkaSink(broker, topic)
        sink.emit([book("1")])
        sink.flush()

        source = KafkaSource(broker, [topic], f"g-{uuid.uuid4().hex[:8]}", max_idle_polls=8)
        try:
            received = drain(source, 1)
        finally:
            source.close()

        assert len(received) == 1
        assert isinstance(received[0], BookEvent)
        assert received[0].source_id == "1"

    def test_the_envelope_survives_the_wire(self, broker: str, topic: str) -> None:
        # Serialisation is the one thing a fake producer cannot vouch for.
        sink = _sink(broker, topic)
        original = book("42")
        sink.emit([original])
        sink.flush()

        source = _source(broker, topic)
        try:
            received = drain(source, 1)[0]
        finally:
            source.close()

        assert received == original


class TestPartitioning:
    def test_one_book_always_lands_on_one_partition(self, broker: str, topic: str) -> None:
        # The ordering guarantee: every update to a book keys the same way, so
        # they cannot be processed out of order across partitions.
        from confluent_kafka import Consumer

        sink = _sink(broker, topic)
        sink.emit([book("same") for _ in range(6)])
        sink.flush()

        consumer = Consumer(
            {
                "bootstrap.servers": broker,
                "group.id": f"p-{uuid.uuid4().hex[:8]}",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        consumer.subscribe([topic])
        partitions = set()
        for _ in range(40):
            message = consumer.poll(1.0)
            if message is not None and message.error() is None:
                partitions.add(message.partition())
            if len(partitions) > 1:
                break
        consumer.close()

        assert len(partitions) == 1

    def test_a_marker_lands_on_the_partition_it_names(self, broker: str, topic: str) -> None:
        # Markers are addressed, not hashed: the barrier writes exactly one to
        # each partition, and a hashed marker would leave partitions unclosed.
        from confluent_kafka import Consumer

        sink = _sink(broker, topic)
        sink.emit(
            # The marker names the *logical* topic whose boundary it closes;
            # markers are restricted to books.raw/books.clean by design. The
            # physical topic is randomised only to isolate this test.
            [
                PartitionMarker(run_id=RUN_ID, topic="books.raw", partition=n)
                for n in range(PARTITIONS)
            ]
        )
        sink.flush()

        consumer = Consumer(
            {
                "bootstrap.servers": broker,
                "group.id": f"m-{uuid.uuid4().hex[:8]}",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        consumer.subscribe([topic])
        landed = {}
        for _ in range(60):
            message = consumer.poll(1.0)
            if message is None or message.error() is not None:
                continue
            from pipeline.models.events import decode_event

            event = decode_event(message.value())
            if isinstance(event, PartitionMarker):
                landed[event.partition] = message.partition()
            if len(landed) == PARTITIONS:
                break
        consumer.close()

        assert landed == {n: n for n in range(PARTITIONS)}


class TestAtLeastOnce:
    def test_an_uncommitted_offset_is_redelivered(self, broker: str, topic: str) -> None:
        """The guarantee the whole design rests on.

        A consumer that reads an event and dies before committing must see it
        again. That is what makes "commit after the database write" safe, and
        it is the behaviour no fake can demonstrate.
        """
        group = f"redeliver-{uuid.uuid4().hex[:8]}"
        sink = _sink(broker, topic)
        sink.emit([book("1")])
        sink.flush()

        first = _source(broker, topic, group)
        received = drain(first, 1)
        first.close()  # closed without committing, as a crash would leave it
        assert len(received) == 1

        second = _source(broker, topic, group)
        try:
            again = drain(second, 1)
        finally:
            second.close()

        assert len(again) == 1
        assert again[0].source_id == "1"

    def test_a_committed_offset_is_not_redelivered(self, broker: str, topic: str) -> None:
        group = f"commit-{uuid.uuid4().hex[:8]}"
        sink = _sink(broker, topic)
        sink.emit([book("1")])
        sink.flush()

        first = _source(broker, topic, group)
        drain(first, 1)
        first.commit()
        first.close()

        second = _source(broker, topic, group)
        try:
            again = drain(second, 1)
        finally:
            second.close()

        assert again == []


def _sink(broker: str, topic: str) -> Any:
    from pipeline.messaging.kafka import KafkaSink

    return KafkaSink(broker, topic)


def _source(broker: str, topic: str, group: str | None = None) -> Any:
    from pipeline.messaging.kafka import KafkaSource

    return KafkaSource(broker, [topic], group or f"g-{uuid.uuid4().hex[:8]}", max_idle_polls=8)
