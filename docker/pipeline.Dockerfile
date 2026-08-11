# The pipeline package, installed from the lockfile.
#
# Multi-stage so uv and the build toolchain never reach the runtime image: the
# thing that ships should contain the dependencies and nothing that installed
# them.

FROM python:3.12-slim-bookworm AS build

# uv from PyPI at a pinned version rather than its own image: one registry
# instead of two, and the version is visible here rather than in a tag.
ARG UV_VERSION=0.12.3
RUN pip install --no-cache-dir "uv==${UV_VERSION}"

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer. They change far less often than the
# source, so an edit to a module does not re-resolve the whole environment.
# README too: pyproject declares it, so the build fails without it.
COPY pyproject.toml uv.lock README.md ./
# --extra kafka: the consumer services need the client, and it is an optional
# extra so that a phase 1 install stays light. Omitting it here builds an image
# whose consumers crash on their first import — which no unit test catches,
# because they run against a fake client.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra kafka

COPY src/ ./src/
COPY migrations/ ./migrations/
COPY alembic.ini ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra kafka


FROM python:3.12-slim-bookworm AS runtime

# Non-root: nothing here needs to write outside the staging volume, and a
# container that cannot escalate is one fewer thing to reason about.
RUN groupadd --system pipeline \
    && useradd --system --gid pipeline --create-home pipeline

WORKDIR /app

COPY --from=build --chown=pipeline:pipeline /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER pipeline

ENTRYPOINT ["pipeline"]
CMD ["--help"]
