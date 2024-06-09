import json
import pickle
from http import HTTPStatus

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from mlops_team_project.src.predict import predict

app = FastAPI()

with open("models/xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)


class PredictRequest(BaseModel):
    data: str


@app.get("/")
def root():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/predict/")
def predict_fastapi(predict_request: PredictRequest):

    try:
        data_dict = json.loads(predict_request.data)

        print(f"payload received = {data_dict}")

        return predict(data_dict, model)
    except Exception as e:
        return {"error": None}
