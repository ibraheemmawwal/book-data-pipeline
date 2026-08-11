"""Command-line entry point for the pipeline package.

The ingestion command lands with the load layer. Until then the installed
entry point remains honest and useful: it can report its version and enumerate
the release's available commands instead of importing a module that does not
exist.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pipeline import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI without parsing process-global arguments."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Ingest and canonicalise public book metadata.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the currently available command surface."""
    parser = build_parser()
    parser.parse_args(list(argv) if argv is not None else None)
    parser.print_help()
    return 0
