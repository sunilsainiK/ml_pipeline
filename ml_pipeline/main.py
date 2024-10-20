import os
import logging
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from api.prediction import router as prediction_router
from api.training import router as training_router
from core.preprocessing import load_and_preprocess, preprocess_input  # Added preprocess_input
from core.training import train_model

# Setting up logging configuration
LOG_DIR = './logs/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(level=logging.INFO, filename=os.path.join(LOG_DIR, 'app.log'),
                    filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables for model and data paths
DATA_PATH = os.getenv("DATA_PATH", "Data/datatraining.txt")
PREPROCESSED_DATA_PATH = "./models/preprocessed_data.pkl"
MODEL_PATH = "./models/xgboost_model.pkl"

# Load the trained model (this assumes the model is already trained and saved)
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

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
        await start_training_process()    # Train the model after preprocessing
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
async def predict(data: dict):
    """
    Endpoint to make predictions.
    Expects a JSON object with features, including the 'date' for time-based features.
    """
    try:
        # Preprocess the input data using the newly added preprocess function
        features_array = preprocess_input(data)

        # Load the model
        model = load_model()

        # Make the prediction
        prediction = model.predict(features_array)

        return {"prediction": prediction.tolist()}  # Convert numpy array to list for JSON serialization

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

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
