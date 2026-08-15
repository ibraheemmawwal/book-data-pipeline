"""Environment-driven configuration.

``Settings`` is a pure function of the process environment. It reads no
``.env`` file of its own, so a value is either explicitly passed, exported by
the caller, or the documented default — which is what makes both tests and
container runs predictable. Load a local file explicitly instead:

    uv run --env-file .env pipeline ingest

Two rules are encoded here rather than left to the extractors, because both are
usage-policy commitments rather than tuning knobs: Open Library is capped at one
request per second and requires an identifying contact address.

Google Books is credential-gated. A missing key must produce an observable skip
recorded in ``source_runs``, never a crash and never a silent no-op, so that a
clean clone starts without credentials.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Self

import structlog
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pipeline import __version__
from pipeline.models.domain import SourceName

__all__ = ["Settings", "SourceName"]

logger = structlog.get_logger(__name__)

REPOSITORY_URL = "https://github.com/ibraheemmawwal/book-data-pipeline"

# Addresses that look like a contact and are not one.
PLACEHOLDER_CONTACTS = frozenset({"you@example.com", "user@example.com", "test@example.com"})

# The load stage commits one transaction per batch; the TRD caps that at 1,000
# records so a failure never rolls back an unbounded amount of work.
MAX_LOAD_BATCH_SIZE = 1000
SHA256_HEX_LENGTH = 64

# Politeness cap for Open Library. Deliberately a ceiling, not a default.
MAX_OPENLIBRARY_REQUESTS_PER_SECOND = 1.0

# Ceiling for the unofficial Goodreads integration. Politeness toward a source
# whose terms restrict automated collection is not a tuning knob.
MAX_GOODREADS_REQUESTS_PER_SECOND = 5.0

ENV_PREFIX = "PIPELINE_"

_ACCEPTED_DATABASE_SCHEMES = ("postgresql://", "postgresql+psycopg://")

# Extraction order, which is not the canonical-priority order in SourceName.
# Gutendex is the bulk source and runs first; the other two enrich.
# Resolution order, which is not the SourceName declaration order. Goodreads
# resolves first; the documented APIs fill gaps; Gutendex is a last resort.
_EXTRACTION_ORDER = (
    SourceName.GOODREADS,
    SourceName.OPENLIBRARY,
    SourceName.GOOGLEBOOKS,
    SourceName.GUTENDEX,
)


class Settings(BaseSettings):
    """Runtime configuration for every stage of the pipeline."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        # A typo'd PIPELINE_* variable that silently does nothing is a
        # debugging trap, so unknown ones fail at startup.
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    # --- Catalogue database -------------------------------------------------
    database_url: str

    # --- Staging ------------------------------------------------------------
    staging_dir: Path = Path("./staging")
    staging_retention_days: Annotated[int, Field(ge=0)] = 7

    # --- Load ---------------------------------------------------------------
    load_batch_size: Annotated[int, Field(ge=1, le=MAX_LOAD_BATCH_SIZE)] = MAX_LOAD_BATCH_SIZE

    # --- HTTP ---------------------------------------------------------------
    http_connect_timeout_seconds: Annotated[float, Field(gt=0)] = 5.0
    http_read_timeout_seconds: Annotated[float, Field(gt=0)] = 30.0
    http_max_attempts: Annotated[int, Field(ge=1)] = 5

    # --- Gutendex -----------------------------------------------------------
    gutendex_enabled: bool = True
    gutendex_base_url: str = "https://gutendex.com"
    gutendex_max_records: Annotated[int, Field(ge=1)] = 6000
    # Small on purpose: Gutendex is a last resort, and a Goodreads outage must
    # not quietly promote it back to being the bulk source.
    gutendex_max_last_resort_queries_per_run: Annotated[int, Field(ge=0)] = 200

    # --- Open Library -------------------------------------------------------
    openlibrary_enabled: bool = True
    openlibrary_base_url: str = "https://openlibrary.org"
    openlibrary_contact_email: str | None = None
    openlibrary_requests_per_second: Annotated[
        float, Field(gt=0, le=MAX_OPENLIBRARY_REQUESTS_PER_SECOND)
    ] = MAX_OPENLIBRARY_REQUESTS_PER_SECOND
    openlibrary_max_records: Annotated[int, Field(ge=1)] = 500
    # Budget exhaustion is an observable skip, not permission to bulk-page a
    # source whose guidance points at dumps for volume.
    openlibrary_max_fallback_queries_per_run: Annotated[int, Field(ge=0)] = 500

    # How many candidates may be resolved at once. Each source's published rate
    # is enforced separately by a limiter held for the run, so this bounds how
    # much latency overlaps rather than how fast any source is asked. Eight is
    # enough to keep a 1/second source saturated when a call takes ~3 seconds;
    # higher just queues on the limiter.
    resolution_concurrency: Annotated[int, Field(ge=1, le=32)] = 8

    # --- Candidate discovery ------------------------------------------------
    # A pinned dump and its digest. Discovery is only reproducible against a
    # known input, so an unpinned checksum is allowed but is a deliberate
    # decision to give that up rather than an oversight.
    openlibrary_dump_path: Path | None = None
    openlibrary_dump_sha256: str | None = None
    discovery_manifest_path: Path = Path("./staging/candidates.jsonl")
    discovery_languages: str = "eng"
    discovery_max_candidates: Annotated[int, Field(ge=1)] = 20000

    # --- Google Books -------------------------------------------------------
    googlebooks_enabled: bool = True
    googlebooks_base_url: str = "https://www.googleapis.com"
    googlebooks_api_key: SecretStr | None = None
    googlebooks_max_records: Annotated[int, Field(ge=1)] = 500
    # Kept below the cloud project quota so one upstream outage cannot spend a
    # whole day's allowance in minutes.
    # Off by default. When set, documented sources are queried even for a
    # candidate something already resolved — which is the only way a book ends
    # up with more than one source, and therefore the only way a cross-source
    # disagreement can exist to be reported. It costs one query per candidate
    # per source, bounded by the same per-run budgets, so it is a deliberate
    # spend rather than a default.
    # How much of the published dump to pull. None fetches all ~12 GB, which
    # is right for a real deployment and absurd for a demo; a slice is genuine
    # dump records either way.
    dump_fetch_max_lines: int | None = None

    # Contested re-resolution runs as part of the pipeline. It still refuses
    # unless the Goodreads gates are set, so this being on changes when the
    # decision is checked, not whether it is.
    # Whether ingestion consults Goodreads for every candidate.
    #
    # Off by default, and that is the substantive position rather than a
    # cautious one: its terms restrict automated collection, so asking it about
    # every book cannot be justified by the fact that some books need
    # adjudicating. The adapter stays wired for targeted use — contested
    # resolution today, search later — and this decides only whether the bulk
    # path uses it.
    goodreads_in_resolution: bool = False

    resolve_contested_enabled: bool = True
    contested_min_conflicts: Annotated[int, Field(ge=1)] = 2
    contested_max_per_run: Annotated[int, Field(ge=0)] = 25

    # How many imported Goodreads records one enrichment run completes. Small
    # and hourly: the backlog is finite and not urgent, and a slice that clears
    # ten thousand in two days never looks like a crawl.
    # Records completed per enrichment run.
    #
    # 200 was calibrated when Goodreads was being asked five times a second and
    # the worry was volume. At one request every two seconds the request rate
    # is itself the restraint, and the slice only decides how much of the hour
    # between runs gets used: 200 records is about ten minutes of it, which
    # leaves a 10,000-record backlog fifty hours away.
    #
    # 100, paired with an hourly run.
    #
    # A record needs 1.36 requests of its own — measured over 1,089 enriched:
    # 700 wanted only the book page, 389 also wanted the editions page because
    # the first withheld an ISBN or a year — and Goodreads' own 503s, running
    # at a quarter to a third of requests, push the real figure to about 2.2
    # once retries are counted. At five seconds apart that is roughly 11
    # seconds a record.
    #
    # So 100 records is about 18 minutes, and a block ladder can add seven on
    # top: 25 minutes against a 50-minute timeout inside an hourly interval.
    # The margin is deliberately much wider than the arithmetic needs, because
    # the spacing this rests on is a single 16-request sample and the previous
    # one aged badly. A run that overruns its timeout is the failure mode that
    # has already bitten twice.
    #
    # The next real run is the number to trust over this one. If 11 seconds
    # holds, the backlog clears in about four days and the slice can go up.
    enrich_max_per_run: Annotated[int, Field(ge=0)] = 100

    enrich_from_documented_sources: bool = False

    googlebooks_max_fallback_queries_per_run: Annotated[int, Field(ge=0)] = 500

    # --- Goodreads (unofficial; see the ADR) --------------------------------
    # Both gates default to false. Goodreads ended public API access in 2020,
    # so this integration reads undocumented web contracts under an explicitly
    # accepted risk. A clean clone must run the documented-API path without it,
    # and enabling it has to be a deliberate act rather than a default.
    goodreads_enabled: bool = False
    goodreads_unofficial_source_accepted: bool = False
    goodreads_base_url: str = "https://www.goodreads.com"
    # One request every five seconds.
    #
    # Thirty came from a spacing walk taken while a block was in progress:
    #
    #     2s  ->  8 of 12, then blocked
    #     5s  ->  8 of 12, then blocked
    #    15s  ->  7 of 18: eight straight successes, then blocked and stayed
    #    30s  -> 13 of 13, never blocked
    #
    # Read as a rolling budget of about eight requests. Re-run days later, the
    # same walk does not reproduce: 16 consecutive requests at five seconds
    # returned zero blocks, and 16 at thirty returned zero. What differs
    # between those two sets is only the 503 rate — 6 of 16 against 4 of 16,
    # which at this sample size is not a difference at all.
    #
    # So the block is episodic, not a standing budget, and thirty seconds was
    # paying for it around the clock. Five is also more conservative than the
    # scraper that collected these same pages at two seconds and never saw a
    # block.
    #
    # What makes five safe to hold is that nothing here depends on the block
    # being gone: the escalating waits, the breaker and the cross-run cooldown
    # all still stand, and they are what turn a block that returns into a
    # pause rather than a failed run. If it returns often, the walk is cheap to
    # repeat — and the ceiling above still caps anything faster.
    goodreads_requests_per_second: Annotated[
        float, Field(gt=0, le=MAX_GOODREADS_REQUESTS_PER_SECOND)
    ] = 1.0 / 5.0
    goodreads_max_in_flight: Annotated[int, Field(ge=1, le=1)] = 1
    # Hard timeout: an unofficial contract must never hold a run open.
    goodreads_timeout_seconds: Annotated[float, Field(gt=0, le=30)] = 5.0
    goodreads_circuit_failure_threshold: Annotated[int, Field(ge=1)] = 5
    # Retries for a transient failure — 5xx or a dropped connection — before it
    # counts against the breaker at all. Goodreads serves 503 to about a
    # quarter of requests while answering the rest in full, so without this an
    # ordinary wobble reads as a block.
    #
    # Two, not three. The third retry is the expensive one and it almost never
    # earns its place: measured at 30s spacing, the 503s scatter rather than
    # cluster — 4 of 16, at positions 1, 3, 10 and 11 — which is a source
    # failing at random, not one refusing us in runs. Against random failure a
    # third attempt recovers a further ~1.5% of records and costs every
    # already-doomed record another 30 seconds. Two attempts after the first
    # already clear about 98%.
    #
    # A record that fails all three stays pending and is picked up by a later
    # run, which is the cheaper place to spend the attempt.
    goodreads_transient_retries: Annotated[int, Field(ge=0, le=10)] = 2
    # Waiting out a block, rather than ending the run over it.
    #
    # The block is global — the minute the backlog's records returned 202 and
    # zero bytes, so did Dune and The Catcher in the Rye — so moving to the
    # next book cannot help. But it lifts on its own: probing one page a
    # minute, it cleared between the fourth and fifth. Five minutes is that
    # measurement with a little margin.
    #
    # This is the *ceiling* on one wait, not every wait. The waits escalate —
    # ten seconds, a minute, then two — because "blocked" covers two
    # different events: measured at five seconds between requests, the first
    # was refused and the next succeeded, while a bad one took between four and
    # five minutes to lift. A flat five minutes paid the worst case every time.
    #
    # Two minutes, not five, and five waits rather than three — because
    # Airflow kills a task that stops heartbeating for
    # ``scheduler.task_instance_heartbeat_timeout``, which defaults to exactly
    # 300 seconds. A single five-minute wait sat right on that threshold and
    # was killed by it: the run of 2026-08-15T02:47 waited 10s, 60s, then 300s,
    # and was marked failed and SIGKILLed 5m37s into the last one. The
    # mechanism for surviving a block was the thing ending the run.
    #
    # So the total patience is kept and the individual sleeps are cut below the
    # threshold: 10s, 60s, then 120s three times is about seven minutes, which
    # still outlasts the four-to-five-minute block we measured, with no single
    # wait over two minutes.
    #
    # Surviving all five is the evidence that this is not the block we
    # measured, and only then does the run end and the cooldown apply.
    goodreads_block_pause_seconds: Annotated[float, Field(ge=0, le=1800)] = 120.0
    goodreads_block_retries: Annotated[int, Field(ge=0, le=10)] = 5
    # How long every path stays away after Goodreads refuses us.
    #
    # The breaker stops one run; this stops the next one. Airflow gives each
    # task a fresh process, so without a written-down refusal the hourly
    # enrichment rediscovers the same block every hour and the sequence of
    # correct runs behaves like a retry loop.
    #
    # Fifteen minutes, not ninety.
    #
    # Ninety was picked to outlast a block of unknown length, before anyone had
    # measured one. Measured, it cleared in about five minutes — so ninety was
    # eighteen times the evidence, and it showed: a single block cost an hour
    # and a half of every DAG over something that had already lifted.
    #
    # This is now the *second* line of defence. A run waits the block out
    # itself, twice, and only a block that survives ten minutes of waiting
    # reaches here — which is genuinely different from the one we measured and
    # worth a longer pause than the run took. Zero disables the wait entirely.
    goodreads_cooldown_minutes: Annotated[int, Field(ge=0)] = 15
    goodreads_title_cache_ttl_seconds: Annotated[int, Field(ge=0)] = 3600
    goodreads_isbn_cache_ttl_seconds: Annotated[int, Field(ge=0)] = 86400
    goodreads_min_match_score: Annotated[float, Field(ge=0, le=1)] = 0.4

    # --- Kafka (phase 2) ----------------------------------------------------
    # Read at DAG parse time to choose the graph shape. A real setting rather
    # than a bare environment lookup, because Settings rejects unknown
    # PIPELINE_* variables — the guard that catches a misspelled name would
    # otherwise reject a legitimate one.
    kafka_enabled: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_raw_topic: str = "books.raw"
    kafka_clean_topic: str = "books.clean"
    kafka_dlq_topic: str = "books.dlq"
    kafka_topic_partitions: Annotated[int, Field(ge=1)] = 3
    kafka_max_processing_attempts: Annotated[int, Field(ge=1)] = 3

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_prefixed_variables(cls, data: Any) -> Any:
        """Fail on a ``PIPELINE_*`` variable that matches no field.

        ``extra="forbid"`` cannot catch these: the environment source only ever
        reads variables it already has a field for, so a misspelled name is
        invisible to pydantic and silently keeps the default. That is precisely
        the failure this project cannot afford — a mistyped
        ``PIPELINE_LOAD_BATCH_SIZE`` would quietly ship the wrong transaction
        size — so the environment is checked directly.
        """
        known = {f"{ENV_PREFIX}{name.upper()}" for name in cls.model_fields}
        unknown = sorted(
            name
            for name in os.environ
            if name.upper().startswith(ENV_PREFIX) and name.upper() not in known
        )
        if unknown:
            msg = f"unknown {ENV_PREFIX}* environment variables: {', '.join(unknown)}"
            raise ValueError(msg)
        return data

    @field_validator("database_url")
    @classmethod
    def _require_postgresql(cls, value: str) -> str:
        """The schema depends on PostgreSQL-only features.

        ``tsvector`` generated columns, ``pg_trgm`` and ``ON CONFLICT`` are not
        portable, so a non-PostgreSQL URL is a configuration error rather than
        a degraded mode.
        """
        if not value.startswith(_ACCEPTED_DATABASE_SCHEMES):
            accepted = " or ".join(_ACCEPTED_DATABASE_SCHEMES)
            msg = f"database_url must start with {accepted}"
            raise ValueError(msg)
        return value

    @field_validator("openlibrary_dump_sha256")
    @classmethod
    def _check_digest_shape(cls, value: str | None) -> str | None:
        """A malformed digest can never match, so it fails at startup instead.

        Discovering for an hour and then failing verification is a much worse
        way to learn that someone pasted a truncated hash.
        """
        if value is None:
            return None
        digest = value.strip().lower()
        if len(digest) != SHA256_HEX_LENGTH or not all(c in "0123456789abcdef" for c in digest):
            msg = "openlibrary_dump_sha256 must be 64 hexadecimal characters"
            raise ValueError(msg)
        return digest

    @field_validator("googlebooks_api_key", mode="before")
    @classmethod
    def _blank_key_is_absent(cls, value: object) -> object:
        """``PIPELINE_GOOGLEBOOKS_API_KEY=`` means "not configured".

        An empty string would otherwise read as a present-but-invalid key and
        turn a clean skip into a 400 from Google.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _require_openlibrary_contact(self) -> Self:
        """Open Library requires identified requests.

        Anonymous bulk traffic is what gets a source blocked, so the address is
        mandatory whenever the extractor is enabled.
        """
        if self.openlibrary_enabled and not self.openlibrary_contact_email:
            msg = (
                "openlibrary_contact_email is required when openlibrary_enabled is true; "
                "Open Library's usage policy requires identified requests"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _require_an_active_source(self) -> Self:
        """A run with nothing to extract is a misconfiguration, not a no-op."""
        if not self.active_sources():
            names = ", ".join(source.value for source in _EXTRACTION_ORDER)
            msg = f"at least one source must be runnable (one of: {names})"
            raise ValueError(msg)
        return self

    def _is_enabled(self, source: SourceName) -> bool:
        match source:
            case SourceName.GOODREADS:
                return self.goodreads_enabled
            case SourceName.GUTENDEX:
                return self.gutendex_enabled
            case SourceName.OPENLIBRARY:
                return self.openlibrary_enabled
            case SourceName.GOOGLEBOOKS:
                return self.googlebooks_enabled

    def user_agent(self) -> str:
        """The ``User-Agent`` sent to every source.

        Carries the contact address when Open Library is in play, because that
        is the source whose policy requires it.

        A placeholder address is worse than a missing one: it satisfies the
        letter of "identify yourself" while reaching nobody, which is the
        opposite of the point. It is left in the header rather than stripped —
        removing it would hide the misconfiguration from the one party who
        would notice — but it is announced on every construction, because a
        stack that has been quietly rude for a week is the failure here.
        """
        identity = f"book-data-pipeline/{__version__} (+{REPOSITORY_URL}"
        if self.openlibrary_enabled and self.openlibrary_contact_email:
            if self.openlibrary_contact_email in PLACEHOLDER_CONTACTS:
                logger.warning(
                    "config.placeholder_contact_email",
                    contact=self.openlibrary_contact_email,
                    detail=(
                        "Open Library asks to be told who is calling; this address "
                        "reaches nobody. Set PIPELINE_OPENLIBRARY_CONTACT_EMAIL."
                    ),
                )
            identity += f"; {self.openlibrary_contact_email}"
        return identity + ")"

    def discovery_language_set(self) -> frozenset[str] | None:
        """Languages to keep during discovery, or ``None`` for no filter."""
        codes = {c.strip().lower() for c in self.discovery_languages.split(",") if c.strip()}
        return frozenset(codes) or None

    def active_sources(self) -> tuple[SourceName, ...]:
        """Sources that will actually run, in extraction order.

        Gutendex leads because it supplies the bulk of the catalogue; the other
        two are bounded enrichment passes.
        """
        return tuple(source for source in _EXTRACTION_ORDER if self.skip_reason(source) is None)

    def skip_reason(self, source: SourceName) -> str | None:
        """Why ``source`` will not run, or ``None`` if it will.

        The string is written to ``source_runs.error`` so a skipped source is
        visible in the run record instead of being inferred from a gap.
        """
        if not self._is_enabled(source):
            return f"{source.value} is disabled by configuration"
        if source is SourceName.GOOGLEBOOKS and self.googlebooks_api_key is None:
            return "googlebooks is enabled but no API key is configured"
        if source is SourceName.GOODREADS and not self.goodreads_unofficial_source_accepted:
            # Two gates, not one. Enabling an unofficial source has to be a
            # separate, deliberate acknowledgement of the documented risk.
            return (
                "goodreads is enabled but the unofficial-source risk has not been "
                "accepted; set PIPELINE_GOODREADS_UNOFFICIAL_SOURCE_ACCEPTED=true"
            )
        return None
