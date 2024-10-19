# File: ml_pipeline/api/prediction.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ml_pipeline.utils.load_model import load_model
import pandas as pd
import os

router = APIRouter()


class SensorData(BaseModel):
    Temperature: float
    Humidity: float
    Light: float
    CO2: float
    HumidityRatio: float
    hour: int
    day_of_week: int


@router.post("/predict")
def predict(data: SensorData):
    try:
        model_path = os.getenv("MODEL_PATH", "ml_pipeline/models/xgboost_model.pkl")
        model = load_model(model_path)
        input_data = pd.DataFrame([data.dict()])
        prediction = model.predict(input_data)
        return {"occupancy": int(prediction[0])}

    except FileNotFoundError as fnf_error:
        raise HTTPException(status_code=404, detail=f"Model not found: {fnf_error}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
