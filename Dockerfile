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

#login to wandb (need to pass in a cmd arg for the login stuff)
RUN make wandb_login

ENTRYPOINT ["make"]

CMD ["run_model"]


