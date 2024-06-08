FROM python:3.11
LABEL authors="henry"

RUN pip install poetry

RUN apt-get update && apt-get install -y make

ENV PATH="${PATH}:/root/.poetry/bin"

ENV IN_CONTAINER Yes

#replace with actual key
ENV LOG_DIR=/app/logs/logs/
ENV PERF_DIR=/app/logs/profiling

RUN mkdir -p $LOG_DIR $PERF_DIR

WORKDIR /app

COPY . /app

#Install dependencies, except dev dependencies just in case
RUN poetry install --no-dev

EXPOSE 8080

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
