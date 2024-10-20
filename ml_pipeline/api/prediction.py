from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ml_pipeline.utils import load_model
import pandas as pd
import os

# Create an API router for prediction endpoints
router = APIRouter()


# Defining the structure of the input data
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
    """
    This endpoint receives sensor data as input and returns a prediction about room occupancy.
    """
    try:
        # Load the trained model from the specified path
        model_path = os.getenv("MODEL_PATH", "ml_pipeline/models/xgboost_model.pkl")
        model = load_model(model_path)

        # Convert the input data to a pandas DataFrame (model expects tabular data)
        input_data = pd.DataFrame([data.dict()])

        # Make the prediction and return the occupancy result
        prediction = model.predict(input_data)
        return {"occupancy": int(prediction[0])}

    except FileNotFoundError as fnf_error:
        raise HTTPException(status_code=404, detail=f"Model not found: {fnf_error}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
