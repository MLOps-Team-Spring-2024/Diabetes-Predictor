FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install numpy
RUN pip install xgboost

COPY app .
COPY models models
COPY mlops_team_project mlops_team_project

EXPOSE $PORT

CMD exec uvicorn main:app --port $PORT --host 0.0.0.0 --workers 1
