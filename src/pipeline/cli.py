"""Command-line entry point.

``ingest`` is the v0.1 release: one command that discovers candidates,
resolves them through the source hierarchy, canonicalises what comes back and
loads the catalogue, writing a run record either way.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import structlog

from pipeline import __version__
from pipeline.config import Settings
from pipeline.discover import build_manifest

logger = structlog.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI without parsing process-global arguments."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Ingest and canonicalise public book metadata.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")

    discover = commands.add_parser(
        "discover", help="Build a candidate manifest from an Open Library dump."
    )
    discover.add_argument("--dump", type=Path, help="Path to ol_dump_editions_*.txt.gz")
    discover.add_argument("--out", type=Path, help="Manifest path to write.")
    discover.add_argument("--limit", type=int, help="Stop after this many candidates.")

    ingest = commands.add_parser("ingest", help="Resolve candidates and load the catalogue.")
    ingest.add_argument("--limit", type=int, help="Stop after this many candidates.")

    return parser


def _discover(settings: Settings, args: argparse.Namespace) -> int:
    dump = args.dump or settings.openlibrary_dump_path
    if dump is None:
        logger.error("cli.no_dump", hint="pass --dump or set PIPELINE_OPENLIBRARY_DUMP_PATH")
        return 2

    written = build_manifest(
        Path(dump),
        args.out or settings.discovery_manifest_path,
        languages=settings.discovery_language_set(),
        max_candidates=args.limit or settings.discovery_max_candidates,
        expected_sha256=settings.openlibrary_dump_sha256,
    )
    logger.info("cli.discover_complete", candidates=written)
    return 0


def _ingest(settings: Settings, args: argparse.Namespace) -> int:
    # Deferred so `pipeline --version` and `discover` do not pay to import a
    # database driver they never use.
    from pipeline.ingest import run_ingestion  # noqa: PLC0415

    report = run_ingestion(settings, limit=args.limit)
    logger.info(
        "cli.ingest_complete",
        status=report.status,
        candidates=report.candidates,
        resolved=report.resolved,
        books_inserted=report.books_inserted,
        books_unchanged=report.books_unchanged,
        rejected=report.rejected,
    )
    return 0 if report.status != "failed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command is None:
        parser.print_help()
        return 0

    settings = Settings()
    if args.command == "discover":
        return _discover(settings, args)
    return _ingest(settings, args)


if __name__ == "__main__":
    sys.exit(main())
