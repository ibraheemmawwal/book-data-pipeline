"""The Kafka adapters.

These test the guarantees rather than the client: that a sink refuses to report
success the broker never gave, that keys put a book's updates on one partition,
that markers are addressed rather than hashed, and that an undecodable message
cannot block a partition. A real broker is exercised separately in the
integration suite; none of that is needed to pin the behaviour below.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from uuid import uuid4

import pytest

from pipeline.messaging.kafka import (
    CONSUMER_CONFIG,
    PRODUCER_CONFIG,
    DeliveryFailedError,
    KafkaSink,
    KafkaSource,
)
from pipeline.models.domain import SourceName
from pipeline.models.events import BookEvent, EventType, PartitionMarker

RUN_ID = uuid4()


def book(source_id: str = "1", **overrides: Any) -> BookEvent:
    return BookEvent(
        run_id=RUN_ID,
        source=SourceName.GUTENDEX,
        source_id=source_id,
        payload={"id": source_id},
        **overrides,
    )


class FakeProducer:
    """Records produce calls and lets a test choose what delivery does."""

    def __init__(self, config: dict[str, Any], *, fail: str | None = None) -> None:
        self.config = config
        self.produced: list[dict[str, Any]] = []
        self.flushed = 0
        self._fail = fail
        self.undelivered = 0

    def produce(self, topic: str, **kwargs: Any) -> None:
        self.produced.append({"topic": topic, **kwargs})
        callback = kwargs.get("on_delivery")
        if callback is not None:
            callback(self._fail, None)

    def poll(self, _timeout: float) -> None:
        return None

    def flush(self, _timeout: float) -> int:
        self.flushed += 1
        return self.undelivered


class FakeMessage:
    def __init__(self, value: bytes | None, error: str | None = None) -> None:
        self._value, self._error = value, error

    def value(self) -> bytes | None:
        return self._value

    def error(self) -> str | None:
        return self._error

    def offset(self) -> int:
        return 0

    def partition(self) -> int:
        return 0


class FakeConsumer:
    def __init__(self, config: dict[str, Any], messages: list[Any] | None = None) -> None:
        self.config = config
        self.subscribed: list[str] = []
        self.commits = 0
        self.closed = False
        self._messages = list(messages or [])

    def subscribe(self, topics: list[str]) -> None:
        self.subscribed = topics

    def poll(self, _timeout: float) -> Any:
        return self._messages.pop(0) if self._messages else None

    def commit(self, *, asynchronous: bool) -> None:
        assert asynchronous is False
        self.commits += 1

    def close(self) -> None:
        self.closed = True


class TestProducerGuarantees:
    def test_acks_all_and_idempotence_are_on(self) -> None:
        # A broker failure must not silently lose an event, and a retried
        # produce must not append it twice.
        assert PRODUCER_CONFIG["acks"] == "all"
        assert PRODUCER_CONFIG["enable.idempotence"] is True

    def test_a_delivery_timeout_is_bounded(self) -> None:
        assert PRODUCER_CONFIG["delivery.timeout.ms"] > 0

    def test_the_config_reaches_the_producer(self) -> None:
        captured: dict[str, Any] = {}

        def factory(config: dict[str, Any]) -> FakeProducer:
            captured.update(config)
            return FakeProducer(config)

        KafkaSink("broker:9092", "books.raw", producer_factory=factory)

        assert captured["bootstrap.servers"] == "broker:9092"
        assert captured["acks"] == "all"


class TestSinkDelivery:
    def _sink(self, **kwargs: Any) -> tuple[KafkaSink, FakeProducer]:
        holder: dict[str, FakeProducer] = {}

        def factory(config: dict[str, Any]) -> FakeProducer:
            holder["p"] = FakeProducer(config, **kwargs)
            return holder["p"]

        return KafkaSink("b:9092", "books.raw", producer_factory=factory), holder["p"]

    def test_events_are_produced(self) -> None:
        sink, producer = self._sink()
        sink.emit([book("1"), book("2")])

        assert len(producer.produced) == 2

    def test_a_raw_event_keys_on_ingestion_identity(self) -> None:
        # All updates to one book land on one partition and stay ordered.
        sink, producer = self._sink()
        sink.emit([book("1")])

        assert producer.produced[0]["key"] == b"gutendex:1"

    def test_a_clean_event_keys_on_canonical_identity(self) -> None:
        sink, producer = self._sink()
        sink.emit(
            [
                book(
                    "1",
                    event_type=EventType.BOOK_CLEAN,
                    identity_key="isbn:9780553380163",
                )
            ]
        )

        assert producer.produced[0]["key"] == b"isbn:9780553380163"

    def test_a_marker_is_addressed_to_its_partition(self) -> None:
        # Markers are written to a specific partition, not hashed onto one:
        # the barrier writes exactly one to each.
        sink, producer = self._sink()
        sink.emit([PartitionMarker(run_id=RUN_ID, topic="books.raw", partition=2)])

        assert producer.produced[0]["partition"] == 2
        assert "key" not in producer.produced[0]

    def test_flush_succeeds_when_everything_was_delivered(self) -> None:
        sink, _ = self._sink()
        sink.emit([book()])
        sink.flush()

    def test_a_delivery_error_raises_on_flush(self) -> None:
        # A sink that reported success for an event the broker never took
        # would make the run's counts a lie.
        sink, _ = self._sink(fail="broker unreachable")
        sink.emit([book()])

        with pytest.raises(DeliveryFailedError, match="broker unreachable"):
            sink.flush()

    def test_undelivered_events_raise_on_flush(self) -> None:
        sink, producer = self._sink()
        producer.undelivered = 3
        sink.emit([book()])

        with pytest.raises(DeliveryFailedError, match="3 event"):
            sink.flush()

    def test_failures_do_not_persist_across_flushes(self) -> None:
        sink, producer = self._sink(fail="transient")
        sink.emit([book()])
        with pytest.raises(DeliveryFailedError):
            sink.flush()

        producer._fail = None
        sink.emit([book()])
        sink.flush()


class TestConsumerGuarantees:
    def test_auto_commit_is_off(self) -> None:
        # The offset is ours to commit, after the effect has landed.
        assert CONSUMER_CONFIG["enable.auto.commit"] is False

    def test_it_starts_from_the_beginning_of_a_new_topic(self) -> None:
        assert CONSUMER_CONFIG["auto.offset.reset"] == "earliest"

    def _source(self, messages: list[Any]) -> tuple[KafkaSource, FakeConsumer]:
        holder: dict[str, FakeConsumer] = {}

        def factory(config: dict[str, Any]) -> FakeConsumer:
            holder["c"] = FakeConsumer(config, messages)
            return holder["c"]

        source = KafkaSource("b:9092", ["books.raw"], "transform", consumer_factory=factory)
        return source, holder["c"]

    def test_it_subscribes_to_its_topics(self) -> None:
        _, consumer = self._source([])

        assert consumer.subscribed == ["books.raw"]

    def test_it_decodes_events(self) -> None:
        source, _ = self._source([FakeMessage(book("1").to_json())])

        seen = []
        for event in source.consume():
            seen.append(event)
            source.stop()

        assert isinstance(seen[0], BookEvent)
        assert seen[0].source_id == "1"

    def test_it_yields_until_stopped(self) -> None:
        messages = [FakeMessage(book(str(n)).to_json()) for n in range(3)]
        source, _ = self._source(messages)

        seen = []
        for event in source.consume():
            seen.append(event)
            if len(seen) == 3:
                source.stop()

        assert len(seen) == 3

    def test_an_undecodable_message_does_not_block_the_partition(self) -> None:
        # It cannot be retried into correctness, and stopping would hold up
        # every well-formed event behind it.
        source, _ = self._source([FakeMessage(b"{not json"), FakeMessage(book("2").to_json())])

        seen = []
        for event in source.consume():
            seen.append(event)
            source.stop()

        assert len(seen) == 1
        assert isinstance(seen[0], BookEvent)
        assert seen[0].source_id == "2"

    def test_a_message_level_error_is_skipped(self) -> None:
        source, _ = self._source(
            [FakeMessage(None, error="partition EOF"), FakeMessage(book("1").to_json())]
        )

        seen = []
        for event in source.consume():
            seen.append(event)
            source.stop()

        assert len(seen) == 1

    def test_commit_is_synchronous(self) -> None:
        # An async commit could be lost on shutdown, which would replay work
        # that had already landed.
        source, consumer = self._source([])
        source.commit()

        assert consumer.commits == 1

    def test_close_leaves_the_group(self) -> None:
        # Otherwise a rebalance waits on a session timeout after every restart.
        source, consumer = self._source([])
        source.close()

        assert consumer.closed is True


class TestProductionWiring:
    """The branch that builds a real client.

    Exercised with a stand-in module rather than the C library: the point is
    that the wiring reaches the right constructor with the right config, not
    that confluent-kafka works.
    """

    def test_the_sink_builds_a_real_producer_when_no_factory_is_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: dict[str, Any] = {}
        module = types.ModuleType("confluent_kafka")
        module.Producer = lambda config: built.setdefault("config", config)  # type: ignore[attr-defined]
        module.Consumer = lambda config: built.setdefault("config", config)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "confluent_kafka", module)

        KafkaSink("broker:9092", "books.raw")

        assert built["config"]["bootstrap.servers"] == "broker:9092"
        assert built["config"]["enable.idempotence"] is True

    def test_the_source_builds_a_real_consumer_when_no_factory_is_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class StandIn:
            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config
                self.subscribed: list[str] = []

            def subscribe(self, topics: list[str]) -> None:
                self.subscribed = topics

        module = types.ModuleType("confluent_kafka")
        module.Consumer = StandIn  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "confluent_kafka", module)

        source = KafkaSource("broker:9092", ["books.raw"], "transform")

        assert source._consumer.config["group.id"] == "transform"
        assert source._consumer.config["enable.auto.commit"] is False


class TestPollTimeout:
    def test_an_empty_poll_is_not_an_event(self) -> None:
        # poll returns None when nothing arrived within the timeout. Treating
        # that as an event would push None through every downstream stage.
        holder: dict[str, FakeConsumer] = {}

        def factory(config: dict[str, Any]) -> FakeConsumer:
            holder["c"] = FakeConsumer(config, [None, None, FakeMessage(book("1").to_json())])
            return holder["c"]

        source = KafkaSource("b:9092", ["books.raw"], "g", consumer_factory=factory)

        seen = []
        for event in source.consume():
            seen.append(event)
            source.stop()

        assert len(seen) == 1
        assert isinstance(seen[0], BookEvent)
