# Airflow 3 with the pipeline package available to the DAG.
#
# Built on the official image rather than installing Airflow ourselves: it
# already resolves Airflow's own constraints, which is the part that goes wrong
# when you do it by hand.

FROM apache/airflow:3.3.0-python3.12

USER root

# Build tooling for any dependency without a wheel for this platform. Removed
# in the same layer so it never reaches the final image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

WORKDIR /opt/airflow

# The pipeline's own dependencies, resolved against Airflow's installed set so
# pip refuses rather than silently downgrading something Airflow needs.
COPY --chown=airflow:root pyproject.toml uv.lock README.md ./
COPY --chown=airflow:root src/ ./src/
COPY --chown=airflow:root migrations/ ./migrations/
COPY --chown=airflow:root alembic.ini ./

RUN pip install --no-cache-dir -e . \
    && python -c "import pipeline; print('pipeline', pipeline.__version__)"

USER root
RUN apt-get purge -y build-essential && apt-get autoremove -y
USER airflow
