"""The command line.

The CLI is the release's front door: `discover` builds a manifest, `ingest`
runs the pipeline. Both are thin, and the tests keep them that way — anything
worth testing here belongs in the module the command calls.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

import pipeline.ingest as ingest_module
from pipeline import __version__, services
from pipeline.cli import build_parser, main


class TestParser:
    def test_version_is_reported(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])

        assert exit_info.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_no_command_prints_help_and_succeeds(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Running the bare command should explain itself, not fail.
        assert main([]) == 0
        assert "ingest" in capsys.readouterr().out

    def test_both_commands_are_offered(self) -> None:
        parser = build_parser()

        assert "discover" in parser.format_help()
        assert "ingest" in parser.format_help()

    def test_an_unknown_command_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            main(["teleport"])


class TestDiscoverCommand:
    def test_it_writes_a_manifest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dump = _dump(tmp_path)
        out = tmp_path / "manifest.jsonl"
        _configure(monkeypatch, tmp_path)

        assert main(["discover", "--dump", str(dump), "--out", str(out)]) == 0
        assert json.loads(out.read_text().splitlines()[0])["title"] == "Dune"

    def test_a_limit_is_honoured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dump = _dump(tmp_path, count=5)
        out = tmp_path / "manifest.jsonl"
        _configure(monkeypatch, tmp_path)

        main(["discover", "--dump", str(dump), "--out", str(out), "--limit", "2"])

        assert len(out.read_text().splitlines()) == 2

    def test_without_a_dump_it_exits_non_zero_with_a_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exiting 0 having done nothing is the failure mode to avoid.
        _configure(monkeypatch, tmp_path)

        assert main(["discover"]) == 2


def _configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PIPELINE_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/catalogue")
    monkeypatch.setenv("PIPELINE_OPENLIBRARY_CONTACT_EMAIL", "cli@example.com")
    monkeypatch.setenv("PIPELINE_DISCOVERY_MANIFEST_PATH", str(tmp_path / "default.jsonl"))


def _dump(tmp_path: Path, count: int = 1) -> Path:
    rows = []
    for n in range(count):
        document: dict[str, Any] = {
            "key": f"/books/OL{n}M",
            "title": "Dune" if n == 0 else f"Book {n}",
            "isbn_13": ["9780441172719"],
        }
        rows.append(
            "\t".join(
                [
                    "/type/edition",
                    f"/books/OL{n}M",
                    "1",
                    "2026-01-01T00:00:00.000000",
                    json.dumps(document),
                ]
            )
        )

    path = tmp_path / "dump.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write("\n".join(rows) + "\n")
    return path


class TestIngestCommand:
    def test_it_reports_the_run_and_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _configure(monkeypatch, tmp_path)
        _stub_ingestion(monkeypatch, status="success")

        assert main(["ingest"]) == 0

    def test_a_failed_run_returns_non_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The exit code is what a scheduler acts on, so it has to be honest.
        _configure(monkeypatch, tmp_path)
        _stub_ingestion(monkeypatch, status="failed")

        assert main(["ingest"]) == 1

    def test_a_partial_run_is_not_treated_as_a_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _configure(monkeypatch, tmp_path)
        _stub_ingestion(monkeypatch, status="partial_success")

        assert main(["ingest"]) == 0

    def test_the_limit_reaches_the_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _configure(monkeypatch, tmp_path)
        seen = _stub_ingestion(monkeypatch, status="success")

        main(["ingest", "--limit", "7"])

        assert seen["limit"] == 7


def _stub_ingestion(monkeypatch: pytest.MonkeyPatch, *, status: str) -> dict[str, Any]:
    """Replace the run with a stub; orchestration is tested against a database."""
    seen: dict[str, Any] = {}

    class _Report:
        candidates = resolved = books_inserted = books_unchanged = rejected = 0

        @property
        def status(self) -> str:
            return status

    def fake_run(_settings: Any, *, limit: int | None = None, **_: Any) -> _Report:
        seen["limit"] = limit
        return _Report()

    monkeypatch.setattr(ingest_module, "run_ingestion", fake_run)
    return seen


class TestConsumerCommands:
    """The v2.0 entry points.

    Thin by design: each builds a service from settings and runs it, so the
    tests check the wiring is reached and the exit code is honest. What the
    services actually do is covered against a database and a broker.
    """

    @pytest.mark.parametrize(
        ("command", "expected"),
        [("transform-consumer", "transform"), ("load-consumer", "load")],
    )
    def test_a_consumer_command_runs_its_service(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        command: str,
        expected: str,
    ) -> None:
        _configure(monkeypatch, tmp_path)
        built: list[str] = []

        class Stub:
            def run(self) -> Any:
                return type("S", (), {"__dict__": {}})()

        monkeypatch.setattr(
            services,
            "build_transform_consumer",
            lambda *_a, **_k: built.append("transform") or Stub(),
        )
        monkeypatch.setattr(
            services,
            "build_load_consumer",
            lambda *_a, **_k: built.append("load") or Stub(),
        )

        assert main([command]) == 0
        assert built == [expected]

    def test_the_barrier_command_emits_a_run_boundary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _configure(monkeypatch, tmp_path)
        seen: dict[str, Any] = {}

        monkeypatch.setattr(
            services,
            "emit_run_boundary",
            lambda _settings, run_id, **_k: (seen.setdefault("run_id", run_id) and 3) or 3,
        )

        run_id = "0a8f4c1e-1111-4222-8333-444455556666"
        assert main(["emit-run-boundary", "--run-id", run_id]) == 0
        assert str(seen["run_id"]) == run_id

    def test_the_barrier_requires_a_run_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Emitting a boundary for an unnamed run would close nothing.
        _configure(monkeypatch, tmp_path)

        with pytest.raises(SystemExit):
            main(["emit-run-boundary"])
