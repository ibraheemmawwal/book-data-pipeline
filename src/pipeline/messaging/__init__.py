"""Event transport.

Two protocols and their implementations. Stages depend on the protocols; only
the wiring picks an implementation.
"""

from __future__ import annotations

from pipeline.messaging.contracts import Sink, Source
from pipeline.messaging.file import Event, FileSink, FileSource

__all__ = ["Event", "FileSink", "FileSource", "Sink", "Source"]
