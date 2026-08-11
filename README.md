# book-data-pipeline

[![CI](https://github.com/ibraheemmawwal/book-data-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/ibraheemmawwal/book-data-pipeline/actions/workflows/ci.yml)

An ETL pipeline that ingests book metadata from documented public APIs, validates and
normalises source records, preserves source provenance, resolves them into canonical
books, and loads a PostgreSQL catalogue designed for search and analytics.

> **Status: in development.** Milestone 3 of 9 — all three extractors and the pure
> transform/identity layer are implemented. The PostgreSQL load and CLI ingest command
> land in milestone 4. Releases are tagged `v0.1` (core ETL), `v1.0` (Airflow 3) and
> `v2.0` (Kafka). This README is expanded at each release.

## Design outline

Two identities carry the whole design:

- **Ingestion identity** — `(source, source_id)`. Always present, so it is the conflict
  target for source records and the reason repeated runs are idempotent.
- **Canonical identity** — a valid ISBN-13 when one exists, otherwise a deterministic
  fallback key derived from normalised title, first author and year.

Gutendex supplies most records and publishes no ISBNs at all, so the ISBN-less path is
the common case rather than an edge case. Provenance is kept in `book_sources` so a book
seen by several providers stays one canonical row without discarding where it came from.

## Quickstart

```bash
docker compose up -d
open http://localhost:8080        # Airflow, admin/admin
```

That is the whole setup. Both databases are created and migrated by init
services — there is no manual SQL — and the stack needs no credentials: Google
Books skips observably without a key, and Goodreads stays off unless two
explicit flags are set.

Trigger a run:

```bash
docker compose exec airflow-scheduler airflow dags trigger book_ingestion
```

It ships with a tiny captured dump sample so this works immediately. For a real
run, download an `ol_dump_editions_*.txt.gz` and point at it:

```bash
PIPELINE_OPENLIBRARY_DUMP_PATH=/path/to/ol_dump_editions_2026-01-01.txt.gz \
PIPELINE_OPENLIBRARY_DUMP_SHA256=<digest> docker compose up -d
```

The two databases are deliberately separate. Airflow's metadata database is
Airflow's business, and its schema belongs to whichever Airflow version is
running; putting the catalogue in there would couple a data migration to an
Airflow upgrade and hand the scheduler write access to the thing it is meant to
be orchestrating.

### Phase 2: Kafka

```bash
PIPELINE_KAFKA_ENABLED=true docker compose --profile kafka up -d
```

The DAG's job narrows: it resolves candidates onto `books.raw`, emits the run
boundary and finishes. Transform and load become long-running consumers, so a
slow load no longer holds an Airflow task open for hours.

Which graph Airflow builds is decided when it parses the DAG file, because that
is when the task graph is fixed — a run cannot choose a phase. The flag is set
on the kafka profile and nowhere else, so a default clone always gets phase 1.

Delivery is at-least-once with effectively-once database effects. Offsets are
committed only after the database transaction, so a crash between them replays
the record; the load layer keys on `(source, source_id)` and compares a content
hash, so replaying changes nothing.

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed by uv itself.

```bash
uv sync --all-groups          # create .venv and install everything
uv run ruff check .           # lint
uv run ruff format --check .  # formatting
uv run mypy src/              # strict type checking
uv run pytest                 # everything
uv run pytest -m "not integration"   # fast: no containers
uv run pytest -m integration         # container-backed, needs Docker
```

### Continuous integration

Four jobs run on every push and pull request: `quality` (lint, format, strict
types), `unit`, `postgres-integration`, and `coverage`.

Coverage is measured per test job and gated once, in a job that combines them.
Gating on the unit job alone would measure the wrong thing — the load layer is
exercised almost entirely by container-backed tests, so a unit-only gate is
satisfied by deleting integration-heavy code and failed by writing it. Unit
tests alone score 90%; combined, the suite is at 98%, which is where the gate
sits.

Three further jobs from the TRD (`kafka-integration`, `dag`, `image`) arrive
with the releases that give them something to run. A green job that asserts
nothing is worse than an absent one.

Copy `.env.example` to `.env` for local runs, then pass it explicitly:

```bash
uv run --env-file .env pytest
```

`Settings` deliberately does not read `.env` by itself. Configuration is a pure function
of the process environment, which keeps tests hermetic and matches how Compose and Cloud
Run supply values. A `PIPELINE_*` variable that matches no field is rejected at startup —
a misspelled name that silently keeps a default is the failure this project can least
afford.

### Secret scanning

```bash
uv run pre-commit install
```

`gitleaks` and `detect-private-key` run on every commit. Enable GitHub secret scanning and
push protection on the repository as well; the local hook catches a secret before it
leaves the machine, and push protection catches whatever the hook misses.

### Source credentials

Gutendex and Open Library need no credentials. Open Library requests are identified with
a `User-Agent` containing a contact email and are capped at one request per second.
Google Books is gated on `PIPELINE_GOOGLEBOOKS_API_KEY`: without it the extractor skips
observably and records the reason, so a clean clone runs credential-free.

## Licence

MIT
