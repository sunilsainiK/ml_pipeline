import os
import logging
import pickle
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from ml_pipeline.api.prediction import router as prediction_router
from ml_pipeline.api.training import router as training_router
from ml_pipeline.core.preprocessing import load_and_preprocess, preprocess_input
from ml_pipeline.core.training import train_model
from ml_pipeline.utils import load_model

LOG_DIR = 'ml_pipeline/logs/app.log'  # Change this to your desired path

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(level=logging.DEBUG,  # Set to DEBUG for more detailed logs
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(LOG_DIR, 'app.log')),
                        logging.StreamHandler()  # Log to console as well
                    ])
logger = logging.getLogger(__name__)

# Environment variables for model and data paths
DATA_PATH = "ml_pipeline/Data/datatraining.txt"
PREPROCESSED_DATA_PATH = "ml_pipeline/models/preprocessed_data.pkl"
MODEL_PATH = "ml_pipeline/models/xgboost_model.pkl"

class PredictionInput(BaseModel):
    date: str
    Temperature: float
    Humidity: float
    Light: float
    CO2: float
    HumidityRatio: float



# Initialize FastAPI application
app = FastAPI()

# Include the prediction and training routers
app.include_router(prediction_router)
app.include_router(training_router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI application...")

    # Check if preprocessed data is outdated or missing
    if is_preprocessed_data_outdated(DATA_PATH, PREPROCESSED_DATA_PATH):
        logger.info("Preprocessed data is outdated or missing. Preprocessing...")
        await preprocess_and_save_data()  # Preprocess the data

        logger.info("Training the model with updated data...")
        await start_training_process()  # Train the model after preprocessing
    else:
        logger.info("Preprocessed data is up-to-date. Checking for new data.")
        if check_for_new_data(DATA_PATH, PREPROCESSED_DATA_PATH):
            logger.info("New data detected. Reprocessing and retraining the model...")
            await preprocess_and_save_data()  # Preprocess and train the model again
            await start_training_process()

    yield  # Required for the lifespan context manager


@app.on_event("startup")
async def startup_event():
    async with lifespan(app):
        pass


@app.get("/")
async def root():
    return {"message": "Welcome to the ML Pipeline API!"}


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


@app.post("/train")
async def train(data: UploadFile = File(...)):
    """
    Endpoint to accept new training data and retrain the model.
    Expects a file upload for the new training data.
    """
    try:
        # Save the uploaded file
        with open(DATA_PATH, 'wb') as f:
            content = await data.read()
            f.write(content)

        logger.info("New training data received. Starting training process...")

        # Reprocess the data and retrain the model
        await preprocess_and_save_data()  # Preprocess the new data
        await start_training_process()  # Train the model with the new data

        return {"message": "Training completed successfully."}

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail="Training failed")


def is_preprocessed_data_outdated(data_path, preprocessed_data_path):
    try:
        if not os.path.exists(preprocessed_data_path):
            logger.warning("No preprocessed data found.")
            return True

        data_mod_time = os.path.getmtime(data_path)
        preprocessed_mod_time = os.path.getmtime(preprocessed_data_path)

        if data_mod_time > preprocessed_mod_time:
            logger.info("Preprocessed data is outdated.")
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking data files: {e}")
        return True


async def preprocess_and_save_data():
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)
        with open(PREPROCESSED_DATA_PATH, 'wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        logger.info("Preprocessed data saved successfully.")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


async def start_training_process():
    try:
        logger.info("Starting training process...")
        train_result = train_model(DATA_PATH)
        logger.info(train_result['message'])
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


def check_for_new_data(data_path, preprocessed_data_path):
    try:
        if not os.path.exists(preprocessed_data_path):
            logger.warning("Preprocessed data not found. New data is available.")
            return True

        data_mod_time = os.path.getmtime(data_path)
        preprocessed_mod_time = os.path.getmtime(preprocessed_data_path)

        if data_mod_time > preprocessed_mod_time:
            logger.info("New data has arrived for processing.")
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking for new data: {e}")
        return True

