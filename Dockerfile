FROM python:3.11
LABEL authors="henry"

RUN pip install poetry

RUN apt-get update && apt-get install -y make

ENV PATH="${PATH}:/root/.poetry/bin"

ENV IN_CONTAINER Yes

#replace with actual key
ENV LOG_DIR=/app/logs
ENV PERF_DIR=/app/performance

RUN mkdir -p $LOG_DIR $PERF_DIR

WORKDIR /app

COPY . /app
#Install dependencies, except dev dependencies just in case
RUN poetry install --no-dev

ENTRYPOINT ["poetry", "run", "python", "mlops_team_project/models/xgboost_model.py"]

#CMD ["run_model"]


