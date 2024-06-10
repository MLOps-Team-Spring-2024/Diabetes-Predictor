FROM python:3.11-slim
LABEL authors="henry"

RUN pip install poetry

RUN apt-get update && apt-get install -y make

ENV PATH="${PATH}:/root/.poetry/bin"

ENV IN_CONTAINER Yes

#replace with actual key
ENV LOG_DIR=/app/logs/logs/
ENV PERF_DIR=/app/logs/profiling

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

ENV PYTHONPATH=/app

RUN mkdir -p $LOG_DIR $PERF_DIR

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-dev --no-root && rm -rf $POETRY_CACHE_DIR

COPY mlops_team_project /app/mlops_team_project

#Install dependencies, except dev dependencies just in case
RUN poetry install --no-dev

ENTRYPOINT ["poetry", "run", "python", "mlops_team_project/models/xgboost_model.py"]
