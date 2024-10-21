from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ml_pipeline.utils import load_model
from ml_pipeline.core.preprocessing import preprocess_input  # Assuming you have this function
import pandas as pd
import os

# Create an API router for prediction endpoints
router = APIRouter()

# Defining the structure of the input data
class SensorData(BaseModel):
    date: str  # Date is passed as a string
    Temperature: float
    Humidity: float
    Light: float
    CO2: float
    HumidityRatio: float


@router.post("/predict")
def predict(data: SensorData):
    """
    This endpoint receives sensor data as input and returns a prediction about room occupancy.
    """
    try:
        # Load the trained model from the specified path
        model_path = "ml_pipeline/models/xgboost_model.pkl"
        model = load_model(model_path)

        # Convert the input data to a dictionary and preprocess it (e.g., extract hour and day_of_week)
        input_dict = data.dict()
        input_data = preprocess_input(input_dict)

        # Make the prediction and return the occupancy result
        prediction = model.predict(input_data)
        return {"occupancy": int(prediction[0])}

    except FileNotFoundError as fnf_error:
        raise HTTPException(status_code=404, detail=f"Model not found: {fnf_error}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
