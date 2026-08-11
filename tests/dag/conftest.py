"""Airflow test configuration.

These are the only tests that need Airflow installed, so they carry the `dag`
marker and are deselected everywhere else. Unit test mode keeps Airflow off a
real metadata database.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Airflow is a heavy, optional dependency. Skipping at collection means the
# other CI jobs can install without it and these tests disappear rather than
# erroring on import — the marker alone would not help, because deselection
# happens after collection has already tried to import the module.
pytest.importorskip("airflow", reason="requires the airflow dependency group")

os.environ.setdefault("AIRFLOW__CORE__UNIT_TEST_MODE", "True")
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
os.environ.setdefault("PIPELINE_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/catalogue")
os.environ.setdefault("PIPELINE_OPENLIBRARY_CONTACT_EMAIL", "dag@example.com")

DAGS_FOLDER = Path(__file__).parent.parent.parent / "dags"

# Airflow puts the dags folder on sys.path when it parses; tests that inspect
# the module directly need the same.
if str(DAGS_FOLDER) not in sys.path:
    sys.path.insert(0, str(DAGS_FOLDER))


def _parse() -> object:
    """Parse the DAGs folder fresh.

    Deliberately not cached across shapes: Airflow fixes the task graph when it
    reads the file, so testing both phases means re-importing the module under
    a different environment.
    """
    # Airflow 3 moved DagBag out of airflow.models and dropped
    # include_examples; AIRFLOW__CORE__LOAD_EXAMPLES now does that job.
    from airflow.dag_processing.dagbag import DagBag

    sys.modules.pop("book_ingestion_dag", None)
    return DagBag(dag_folder=str(DAGS_FOLDER))


@pytest.fixture(scope="session")
def dagbag() -> object:
    """The phase 1 graph, which is what a default clone runs.

    The flag is cleared explicitly rather than assumed absent: a developer with
    PIPELINE_KAFKA_ENABLED exported would otherwise get the phase 2 graph here
    and watch the phase 1 assertions fail for no visible reason.
    """
    previous = os.environ.pop("PIPELINE_KAFKA_ENABLED", None)
    try:
        return _parse()
    finally:
        if previous is not None:
            os.environ["PIPELINE_KAFKA_ENABLED"] = previous


@pytest.fixture
def kafka_dagbag(monkeypatch: pytest.MonkeyPatch) -> object:
    """The phase 2 graph."""
    monkeypatch.setenv("PIPELINE_KAFKA_ENABLED", "true")
    bag = _parse()
    monkeypatch.delenv("PIPELINE_KAFKA_ENABLED", raising=False)
    # Leave the module cache clean so a later session-scoped parse is not
    # served the phase 2 shape.
    sys.modules.pop("book_ingestion_dag", None)
    return bag
