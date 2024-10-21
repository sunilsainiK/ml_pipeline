import json
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pandas as pd
import logging
from pydantic import BaseModel
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define paths
MODEL_PATH = "ml_pipeline/models/xgboost_model.pkl"  # Replace with the actual path to your model

# Define the input model for prediction
class PredictionInput(BaseModel):
    date: str
    Temperature: float
    Humidity: float
    Light: float
    CO2: float
    HumidityRatio: float

# Initialize FastAPI app
app = FastAPI()

# Function to preprocess the input data
def preprocess_input(df):
    # Extracting 'hour' and 'weekday' from the 'date' field
    df['hour'] = pd.to_datetime(df['date']).dt.hour  # Extract hour from the date
    df['weekday'] = pd.to_datetime(df['date']).dt.weekday  # Extract weekday from the date

    features = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'hour', 'weekday']]
    print(features.head())
    return np.array(features)

# Function to load the trained model from file
def load_model(model_path):
    logger.info(f"Loading model from {model_path}...")
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

# Prediction endpoint
@app.post("/predict")
async def predict(data: PredictionInput):
    logger.info("Predict endpoint called with data: %s", data.dict())
    try:
        # Convert input to dict and create DataFrame
        data_dict = data.dict()
        df = pd.DataFrame([data_dict])
        logger.info("Data after creating DataFrame: %s", df)

        # Preprocess input
        features_array = preprocess_input(df)
        logger.info("Data after preprocessing: %s", features_array)

        # Load model
        logger.info("Loading the trained model...")
        model = load_model(MODEL_PATH)

        # Make prediction
        logger.info("Making predictions...")
        prediction = model.predict(features_array)
        logger.info("Prediction result: %s", prediction)

        return {"prediction": int(prediction[0])}
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")

# Test client to simulate a request to the FastAPI app
client = TestClient(app)

# Sample input data as a Python dictionary (can also come from a file)
data = {
    "date": "2015-02-04 17:51:00",
    "Temperature": 23.18,
    "Humidity": 27.272,
    "Light": 426,
    "CO2": 721.25,
    "HumidityRatio": 0.00479298817650529
}

# Making the prediction request and print result
response = client.post("/predict", json=data)
print("Prediction Result:", response.json())
