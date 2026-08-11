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


@pytest.fixture(scope="session")
def dagbag() -> object:
    """The project's DAGs, parsed the way Airflow parses them."""
    # Airflow 3 moved DagBag out of airflow.models and dropped
    # include_examples; AIRFLOW__CORE__LOAD_EXAMPLES now does that job.
    from airflow.dag_processing.dagbag import DagBag

    return DagBag(dag_folder=str(DAGS_FOLDER))
