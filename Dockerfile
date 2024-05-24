FROM python:3.11
LABEL authors="henry"

RUN pip install poetry

RUN apt-get update && apt-get install -y make

ENV PATH="${PATH}:/root/.poetry/bin"

#replace with actual key
ENV WANDB_API_KEY = key

WORKDIR /app

COPY . .
#Install dependencies, except dev dependencies just in case
RUN poetry install --no-dev

ENTRYPOINT ["poetry", "run", "python", "mlops_team_project/models/xgboost_model.py"]

#CMD ["run_model"]


