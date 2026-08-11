"""Kafka source and sink: the v2.0 implementation of the same contracts.

Everything here exists to make one sentence true: *at-least-once delivery with
effectively-once database effects*. Not exactly-once — that would need
transactional writes spanning Kafka and PostgreSQL, and the cost of coordinating
two systems buys nothing the idempotent load layer does not already give.

The two halves of that sentence:

**At-least-once.** ``enable.auto.commit=false``, and an offset is committed only
after the downstream effect has succeeded. A crash between the effect and the
commit means redelivery, which is the safe direction to fail in — the unsafe
direction is committing first and losing the record.

**Effectively-once effects.** The load layer keys on ``(source, source_id)`` and
compares a content hash, so processing the same event twice changes nothing.
That is what makes redelivery boring rather than corrupting.

``confluent-kafka`` is a blocking C client. That is deliberate — it is what runs
in production — and it is why these adapters are synchronous while the
extractors are async.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import structlog

from pipeline.models.events import BookEvent, PartitionMarker, decode_event

logger = structlog.get_logger(__name__)

Event = BookEvent | PartitionMarker

# How long a poll waits before returning nothing. Short enough that shutdown is
# responsive, long enough not to spin.
POLL_TIMEOUT_SECONDS = 1.0

# Producer settings that are not tuning knobs.
#
# acks=all           a write is not acknowledged until every in-sync replica
#                    has it, so a broker failure cannot silently lose an event
# enable.idempotence a retried produce is deduplicated by the broker rather
#                    than appended twice
# linger.ms          a small batching window; the throughput difference is
#                    large and the latency cost is irrelevant to a batch run
PRODUCER_CONFIG: dict[str, Any] = {
    "acks": "all",
    "enable.idempotence": True,
    "compression.type": "snappy",
    "linger.ms": 50,
    "delivery.timeout.ms": 120_000,
}

CONSUMER_CONFIG: dict[str, Any] = {
    # The offset is ours to commit, after the effect has landed.
    "enable.auto.commit": False,
    "auto.offset.reset": "earliest",
}


class DeliveryFailedError(Exception):
    """A produce was not acknowledged.

    Raised rather than logged: a sink that reported success for an event the
    broker never took would make the run's counts a lie.
    """


class KafkaSink:
    """Produces events to a topic.

    Keys are set by the event, not the caller: raw events key on ingestion
    identity and clean events on canonical identity, so all updates to one book
    land on one partition and are processed in order.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        *,
        producer_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._topic = topic
        self._failures: list[str] = []

        config = {"bootstrap.servers": bootstrap_servers, **PRODUCER_CONFIG}
        if producer_factory is not None:
            self._producer = producer_factory(config)
        else:
            from confluent_kafka import Producer  # noqa: PLC0415

            self._producer = Producer(config)

    def _on_delivery(self, error: Any, message: Any) -> None:  # noqa: ARG002
        """Record a failed delivery so ``flush`` can raise on it.

        Delivery is asynchronous: without checking the callback, a producer
        reports success the moment it has buffered, not when the broker has
        taken the event.
        """
        if error is not None:
            self._failures.append(str(error))
            logger.error("kafka_sink.delivery_failed", topic=self._topic, error=str(error))

    def emit(self, records: Iterable[Event]) -> None:
        """Buffer events for delivery. Not delivered until ``flush``."""
        for record in records:
            key = record.partition_key() if isinstance(record, BookEvent) else None
            kwargs: dict[str, Any] = {
                "value": record.to_json(),
                "on_delivery": self._on_delivery,
            }
            if key is not None:
                kwargs["key"] = key.encode()
            if isinstance(record, PartitionMarker):
                # Markers are addressed to a partition, not hashed onto one:
                # the barrier writes exactly one to each.
                kwargs["partition"] = record.partition
            self._producer.produce(self._topic, **kwargs)
            self._producer.poll(0)

    def flush(self) -> None:
        """Block until every buffered event is acknowledged.

        Raises:
            DeliveryFailedError: the broker did not take one or more events.
        """
        remaining = self._producer.flush(30)
        if remaining:
            msg = f"{remaining} event(s) still undelivered after flush on {self._topic}"
            raise DeliveryFailedError(msg)
        if self._failures:
            failures, self._failures = self._failures, []
            msg = f"{len(failures)} delivery failure(s) on {self._topic}: {failures[0]}"
            raise DeliveryFailedError(msg)


class KafkaSource:
    """Consumes events from one or more topics until shutdown.

    Unlike the file source this does not end, which is exactly the difference
    the ``Source`` protocol exists to hide from the stages.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topics: list[str],
        group_id: str,
        *,
        consumer_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._topics = topics
        self._running = True

        config = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            **CONSUMER_CONFIG,
        }
        if consumer_factory is not None:
            self._consumer = consumer_factory(config)
        else:
            from confluent_kafka import Consumer  # noqa: PLC0415

            self._consumer = Consumer(config)
        self._consumer.subscribe(topics)

    def consume(self) -> Iterator[Event]:
        """Yield events until ``stop`` is called.

        An undecodable message is skipped rather than fatal — it cannot be
        retried into correctness, and blocking the partition on it would stop
        every well-formed event behind it.
        """
        while self._running:
            message = self._consumer.poll(POLL_TIMEOUT_SECONDS)
            if message is None:
                continue
            if message.error() is not None:
                logger.warning("kafka_source.message_error", error=str(message.error()))
                continue

            try:
                yield decode_event(message.value())
            except Exception as error:
                logger.warning(
                    "kafka_source.undecodable",
                    error=str(error),
                    offset=message.offset(),
                    partition=message.partition(),
                )

    def commit(self) -> None:
        """Commit the current offsets.

        Called only after the downstream effect has succeeded. That ordering is
        the at-least-once guarantee; reversing it would make lost records
        possible in exchange for nothing.
        """
        self._consumer.commit(asynchronous=False)

    def stop(self) -> None:
        """Ask the consume loop to finish after the current poll."""
        self._running = False

    def close(self) -> None:
        """Leave the group cleanly so a rebalance does not wait on a timeout."""
        self._consumer.close()
