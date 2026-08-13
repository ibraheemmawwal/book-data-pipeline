"""Building the services from configuration.

Kept apart from the services themselves so they stay testable with file
adapters and a fake broker. This module is the only place that decides a
consumer talks to Kafka, which is what keeps that decision out of the logic.
"""

from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy import Engine

from pipeline.config import Settings
from pipeline.db import build_engine
from pipeline.messaging.kafka import KafkaSink, KafkaSource
from pipeline.models.events import PartitionMarker
from pipeline.observability.markers import freeze_topology, mark_processing
from pipeline.services.load_consumer import LoadConsumer
from pipeline.services.transform_consumer import TransformConsumer

logger = structlog.get_logger(__name__)

TRANSFORM_GROUP = "book-pipeline-transform"
LOAD_GROUP = "book-pipeline-load"


def build_transform_consumer(settings: Settings, engine: Engine | None = None) -> TransformConsumer:
    """A transform consumer reading books.raw and writing books.clean."""
    active = engine or build_engine(settings.database_url)
    return TransformConsumer(
        active,
        KafkaSource(settings.kafka_bootstrap_servers, [settings.kafka_raw_topic], TRANSFORM_GROUP),
        KafkaSink(settings.kafka_bootstrap_servers, settings.kafka_clean_topic),
        KafkaSink(settings.kafka_bootstrap_servers, settings.kafka_dlq_topic),
        clean_topic=settings.kafka_clean_topic,
        raw_topic=settings.kafka_raw_topic,
        clean_partitions=settings.kafka_topic_partitions,
        max_attempts=settings.kafka_max_processing_attempts,
    )


def build_load_consumer(settings: Settings, engine: Engine | None = None) -> LoadConsumer:
    """A load consumer reading books.clean and writing the catalogue."""
    active = engine or build_engine(settings.database_url)
    return LoadConsumer(
        active,
        KafkaSource(settings.kafka_bootstrap_servers, [settings.kafka_clean_topic], LOAD_GROUP),
        KafkaSink(settings.kafka_bootstrap_servers, settings.kafka_dlq_topic),
        clean_topic=settings.kafka_clean_topic,
    )


def emit_run_boundary(
    settings: Settings,
    run_id: UUID,
    engine: Engine | None = None,
    *,
    records_extracted: int | None = None,
) -> int:
    """Close a run's raw topic: freeze the topology, then emit the markers.

    Order matters. The expectation is written first so a consumer that sees a
    marker always has something to compare it against; emitting first would
    open a window where a marker arrives with no frozen topology and is
    correctly, but uselessly, discarded.

    Partition counts come from configuration rather than from the events, so no
    completion path anywhere contains a literal 3.
    """
    active = engine or build_engine(settings.database_url)
    partitions = settings.kafka_topic_partitions

    with active.begin() as connection:
        freeze_topology(connection, run_id, settings.kafka_raw_topic, partitions)
        freeze_topology(connection, run_id, settings.kafka_clean_topic, partitions)
        mark_processing(connection, run_id, records_extracted=records_extracted)

    sink = KafkaSink(settings.kafka_bootstrap_servers, settings.kafka_raw_topic)
    sink.emit(
        [
            PartitionMarker(run_id=run_id, topic=settings.kafka_raw_topic, partition=n)
            for n in range(partitions)
        ]
    )
    # Flushed here so the barrier fails loudly if the markers did not land;
    # a run whose boundary never arrived would hang in 'processing'.
    sink.flush()

    logger.info("barrier.markers_emitted", run_id=str(run_id), partitions=partitions)
    return partitions
