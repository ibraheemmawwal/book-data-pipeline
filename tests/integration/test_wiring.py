"""Building the services from configuration, and the run barrier.

The barrier is the handover from the DAG to the consumers, and its ordering is
the part worth pinning: the expectation is frozen before any marker is emitted,
so a consumer that sees a marker always has something to compare it against.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import Connection, Engine, insert, select

from pipeline.config import Settings
from pipeline.models.db import ingestion_runs, run_topic_partitions
from pipeline.services import wiring

pytestmark = pytest.mark.integration


class RecordingSink:
    """Stands in for a Kafka sink so the barrier can be tested without a broker."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.emitted: list[Any] = []
        self.flushed = 0

    def emit(self, records: Any) -> None:
        self.emitted.extend(records)

    def flush(self) -> None:
        self.flushed += 1


@pytest.fixture
def settings(migrated_engine: Engine) -> Settings:
    return Settings(  # type: ignore[call-arg]
        database_url=str(migrated_engine.url).replace("***", "test"),
        openlibrary_contact_email="wiring@example.com",
        kafka_topic_partitions=3,
    )


@pytest.fixture
def run_id(engine: Engine) -> uuid.UUID:
    identifier = uuid.uuid4()
    with engine.begin() as connection:
        connection.execute(
            insert(ingestion_runs).values(id=identifier, dag_run_id=f"cli:{identifier}")
        )
    return identifier


class TestRunBoundary:
    def test_it_freezes_topology_for_both_topics(
        self,
        settings: Settings,
        engine: Engine,
        connection: Connection,
        run_id: uuid.UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(wiring, "KafkaSink", RecordingSink)

        wiring.emit_run_boundary(settings, run_id, engine=engine)

        frozen = connection.execute(
            select(run_topic_partitions.c.topic, run_topic_partitions.c.expected_partitions)
            .where(run_topic_partitions.c.run_id == run_id)
            .order_by(run_topic_partitions.c.topic)
        ).all()
        assert [(r.topic, r.expected_partitions) for r in frozen] == [
            ("books.clean", 3),
            ("books.raw", 3),
        ]

    def test_it_emits_one_marker_per_partition(
        self,
        settings: Settings,
        engine: Engine,
        run_id: uuid.UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sinks: list[RecordingSink] = []

        def factory(*args: Any, **kwargs: Any) -> RecordingSink:
            sink = RecordingSink(*args, **kwargs)
            sinks.append(sink)
            return sink

        monkeypatch.setattr(wiring, "KafkaSink", factory)

        assert wiring.emit_run_boundary(settings, run_id, engine=engine) == 3
        assert sorted(m.partition for m in sinks[0].emitted) == [0, 1, 2]

    def test_the_markers_are_flushed(
        self,
        settings: Settings,
        engine: Engine,
        run_id: uuid.UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A run whose boundary never landed would hang in 'processing', so the
        # barrier has to fail loudly rather than return optimistically.
        sinks: list[RecordingSink] = []

        def factory(*_args: Any, **_kwargs: Any) -> RecordingSink:
            sinks.append(RecordingSink())
            return sinks[-1]

        monkeypatch.setattr(wiring, "KafkaSink", factory)

        wiring.emit_run_boundary(settings, run_id, engine=engine)

        assert sinks[0].flushed == 1

    def test_it_hands_the_run_over_to_the_consumers(
        self,
        settings: Settings,
        engine: Engine,
        connection: Connection,
        run_id: uuid.UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The DAG finishes when extraction does; 'processing' is the handover.
        monkeypatch.setattr(wiring, "KafkaSink", RecordingSink)

        wiring.emit_run_boundary(settings, run_id, engine=engine)

        assert (
            connection.execute(
                select(ingestion_runs.c.status).where(ingestion_runs.c.id == run_id)
            ).scalar_one()
            == "processing"
        )

    def test_the_partition_count_comes_from_configuration(
        self,
        settings: Settings,
        engine: Engine,
        connection: Connection,
        run_id: uuid.UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No completion path anywhere contains a literal 3.
        monkeypatch.setattr(wiring, "KafkaSink", RecordingSink)
        five = settings.model_copy(update={"kafka_topic_partitions": 5})

        assert wiring.emit_run_boundary(five, run_id, engine=engine) == 5
        assert (
            connection.execute(
                select(run_topic_partitions.c.expected_partitions).where(
                    run_topic_partitions.c.run_id == run_id,
                    run_topic_partitions.c.topic == "books.raw",
                )
            ).scalar_one()
            == 5
        )

    def test_running_the_barrier_twice_is_harmless(
        self,
        settings: Settings,
        engine: Engine,
        connection: Connection,
        run_id: uuid.UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # An Airflow retry re-runs the task; re-emitting markers is absorbed by
        # the observation primary key downstream.
        monkeypatch.setattr(wiring, "KafkaSink", RecordingSink)

        wiring.emit_run_boundary(settings, run_id, engine=engine)
        wiring.emit_run_boundary(settings, run_id, engine=engine)

        assert connection.execute(
            select(run_topic_partitions.c.topic).where(run_topic_partitions.c.run_id == run_id)
        ).scalars().all() == ["books.raw", "books.clean"]


class TestConsumerConstruction:
    def test_a_transform_consumer_is_built_from_settings(
        self, settings: Settings, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wiring, "KafkaSink", RecordingSink)
        monkeypatch.setattr(wiring, "KafkaSource", RecordingSink)

        consumer = wiring.build_transform_consumer(settings, engine=engine)

        assert consumer._clean_partitions == settings.kafka_topic_partitions
        assert consumer._clean_topic == settings.kafka_clean_topic

    def test_a_load_consumer_is_built_from_settings(
        self, settings: Settings, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(wiring, "KafkaSink", RecordingSink)
        monkeypatch.setattr(wiring, "KafkaSource", RecordingSink)

        consumer = wiring.build_load_consumer(settings, engine=engine)

        assert consumer._clean_topic == settings.kafka_clean_topic

    def test_the_two_consumers_use_different_groups(self) -> None:
        # Sharing a group would have them compete for the same partitions and
        # each see half the events.
        assert wiring.TRANSFORM_GROUP != wiring.LOAD_GROUP
