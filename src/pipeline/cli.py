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
from pipeline.db import build_engine
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

    commands.add_parser(
        "transform-consumer",
        help="Run the books.raw -> books.clean consumer until stopped (v2.0).",
    )
    commands.add_parser(
        "load-consumer",
        help="Run the books.clean -> catalogue consumer until stopped (v2.0).",
    )
    barrier = commands.add_parser(
        "emit-run-boundary",
        help="Freeze topology and emit this run's raw partition markers (v2.0).",
    )
    barrier.add_argument("--run-id", required=True, help="The ingestion run UUID.")

    contested = commands.add_parser(
        "resolve-contested",
        help="Re-resolve books whose sources disagree, through Goodreads.",
    )
    contested.add_argument(
        "--min-conflicts",
        type=int,
        default=2,
        help="How many fields must disagree before a book is worth re-resolving.",
    )
    contested.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Hard cap on books queried. This is the bound that keeps a targeted run targeted.",
    )
    contested.add_argument(
        "--dry-run",
        action="store_true",
        help="List the contested books and make no requests.",
    )
    return parser


def _discover(settings: Settings, args: argparse.Namespace) -> int:
    dump = args.dump or settings.openlibrary_dump_path
    if dump is None:
        logger.error("cli.no_dump", hint="pass --dump or set PIPELINE_OPENLIBRARY_DUMP_PATH")
        return 2

    written, _outcome = build_manifest(
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


def _consume(settings: Settings, which: str) -> int:
    """Run a long-lived consumer until the process is stopped."""
    from pipeline.services import build_load_consumer, build_transform_consumer  # noqa: PLC0415

    build = build_transform_consumer if which == "transform" else build_load_consumer
    stats = build(settings).run()
    logger.info("cli.consumer_stopped", consumer=which, stats=vars(stats))
    return 0


def _emit_boundary(settings: Settings, args: argparse.Namespace) -> int:
    from uuid import UUID  # noqa: PLC0415

    from pipeline.services import emit_run_boundary  # noqa: PLC0415

    partitions = emit_run_boundary(settings, UUID(args.run_id))
    logger.info("cli.boundary_emitted", run_id=args.run_id, partitions=partitions)
    return 0


def _resolve_contested(settings: Settings, args: argparse.Namespace) -> int:
    """Re-resolve contested books, or just list them."""
    from pipeline.contested import find_contested, resolve_contested  # noqa: PLC0415

    if args.dry_run:
        books = find_contested(
            build_engine(settings.database_url),
            minimum_conflicts=args.min_conflicts,
            limit=args.limit,
        )
        for book in books:
            logger.info(
                "contested.candidate",
                title=book["title"][:60],
                conflicts=book["conflicts"],
                sources=book["sources"],
            )
        logger.info("contested.dry_run_complete", books=len(books))
        return 0

    report = resolve_contested(settings, minimum_conflicts=args.min_conflicts, limit=args.limit)
    logger.info("cli.contested_complete", **vars(report))
    return 0


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
    if args.command == "transform-consumer":
        return _consume(settings, "transform")
    if args.command == "load-consumer":
        return _consume(settings, "load")
    if args.command == "resolve-contested":
        return _resolve_contested(settings, args)
    if args.command == "emit-run-boundary":
        return _emit_boundary(settings, args)
    return _ingest(settings, args)


if __name__ == "__main__":
    sys.exit(main())
