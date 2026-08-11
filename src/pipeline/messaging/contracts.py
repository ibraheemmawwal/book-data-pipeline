"""What a stage may assume about where its events come from.

Transform and load are written against these two protocols and nothing else.
In v1.0 a run reads a finite file; in v2.0 it reads a Kafka topic that never
ends. Neither stage can tell, and that is the whole point: the moment a stage
knows which adapter is driving it, the phase-2 change stops being a swap of
implementations and becomes a rewrite of the pipeline.

The protocols are synchronous deliberately. ``confluent-kafka`` is a blocking C
client, and pretending otherwise would mean an async wrapper around a
thread pool for no benefit — the extractors are async because HTTP latency is
worth overlapping; consuming a local socket is not.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, TypeVar, runtime_checkable

T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


@runtime_checkable
class Source(Protocol[T_co]):
    """Somewhere events come from.

    A file source is finite and stops. A Kafka source runs until the service is
    shut down. A stage must work with either without asking which it has.
    """

    def consume(self) -> Iterator[T_co]:
        """Yield events until exhausted, or forever."""
        ...


@runtime_checkable
class Sink(Protocol[T_contra]):
    """Somewhere events go.

    ``emit`` may buffer; only ``flush`` promises the events have left. That
    split is not incidental — a Kafka producer batches, and a caller that never
    flushed would report success for events still sitting in memory.
    """

    def emit(self, records: Iterable[T_contra]) -> None:
        """Hand events to the sink, possibly buffered."""
        ...

    def flush(self) -> None:
        """Block until everything emitted has actually been delivered."""
        ...
