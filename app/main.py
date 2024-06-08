import json
import pickle

from fastapi import FastAPI
from http import HTTPStatus
import numpy as np
from pydantic import BaseModel

app = FastAPI()

with open("models/xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)


class PredictRequest(BaseModel):
    data: str


class PredictionData(BaseModel):
    age: int
    sex: int
    high_chol: int
    chol_check: int
    bmi: float
    smoker: int
    heart_disease: int
    phys_activity: int
    fruits: int
    veggies: int
    hvy_alcohol_consump: int
    gen_hlth: int
    ment_hlth: int
    phys_hlth: int
    diff_walk: int
    stroke: int
    high_bp: int


@app.get("/")
def root():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/predict/")
def predict(predict_request: PredictRequest):

    try:
        data_dict = json.loads(predict_request.data)

        print(f"payload received = {data_dict}")

        prediction_data = PredictionData(**data_dict)

        array_2d = np.array(list(prediction_data.model_dump().values())).reshape(1, -1)

        prediction = model.predict(array_2d)[0]

        return {"prediction": "non-diabetic" if prediction == 0 else "diabetic"}

    except Exception as e:
        return {"error": None}
