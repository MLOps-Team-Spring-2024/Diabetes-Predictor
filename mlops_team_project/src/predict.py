from typing import Dict

import numpy as np
from pydantic import BaseModel
import xgboost


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
    prediction_data = PredictionData(**data_dict)

    array_2d = np.array(list(prediction_data.model_dump().values())).reshape(1, -1)

    prediction = model.predict(array_2d)[0]

    return {"prediction": "non-diabetic" if prediction == 0 else "diabetic"}
