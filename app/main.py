import json
import pickle
from http import HTTPStatus

import numpy as np
from fastapi import FastAPI
from google.cloud import storage
from pydantic import BaseModel

from mlops_team_project.src.predict import predict

app = FastAPI()

client = storage.Client("mlops489-425700")
bucket = client.get_bucket("mlops489-project")
blob = bucket.get_blob("models/xgboost_model.pkl")
model = pickle.loads(blob.download_as_string())


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
