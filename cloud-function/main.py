from typing import Dict
import json
import pickle

import functions_framework
from google.cloud import storage
import numpy as np
from pydantic import BaseModel
import xgboost


@functions_framework.http
def hello_http(request):
    request_data = request.get_json()["data"]
    request_dict = json.loads(request_data)

    print(request_dict)
    print(type(request_dict))

    client = storage.Client()
    bucket = client.get_bucket("mlops489-project")
    blob = bucket.get_blob("models/xgboost_model.pkl")
    model = pickle.loads(blob.download_as_string())

    return predict(request_dict, model)


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


def predict(data_dict: dict, model: xgboost.sklearn.XGBClassifier) -> Dict[str, str]:
    # comment
    prediction_data = PredictionData(**data_dict)

    array_2d = np.array(list(prediction_data.model_dump().values())).reshape(1, -1)

    prediction = model.predict(array_2d)[0]

    return {"prediction": "non-diabetic" if prediction == 0 else "diabetic"}
