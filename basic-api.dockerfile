FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn

COPY basic_fastapi.py basic_fastapi.py

EXPOSE $PORT

CMD exec uvicorn basic_fastapi:app --port $PORT --host 0.0.0.0 --workers 1