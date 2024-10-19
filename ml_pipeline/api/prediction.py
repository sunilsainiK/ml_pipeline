from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from  api.training import router
#from api.training import router
from utils.load_model import load_model
import pandas as pd

# Input data for model prediction


class SensorData(BaseModel):
    Temperature: float
    Humidity: float
    Light: float
    CO2: float
    HumidityRatio: float
    hour: int
    day_of_week: int

router = APIRouter()

@router.post("/predict")
def predict(data: SensorData):
    try:
        #
        model = load_model("ml_pipeline/models/xgboost_model.pkl")
        #convert input data into dataframe
        input_data = pd.DataFrame([data.dict()])

        prediction = model.predict(input_data)
        return {"occupancy":int (prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


