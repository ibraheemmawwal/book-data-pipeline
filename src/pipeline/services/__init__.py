"""Long-running consumer services.

In phase 2 the DAG schedules extraction and finishes; these carry a run from
there. Narrowing Airflow's responsibility is the point, not a workaround for
its task timeout.
"""

from __future__ import annotations

from pipeline.services.load_consumer import LoadConsumer, LoadStats
from pipeline.services.transform_consumer import TransformConsumer, TransformStats
from pipeline.services.wiring import (
    build_load_consumer,
    build_transform_consumer,
    emit_run_boundary,
)

__all__ = [
    "LoadConsumer",
    "LoadStats",
    "TransformConsumer",
    "TransformStats",
    "build_load_consumer",
    "build_transform_consumer",
    "emit_run_boundary",
]
