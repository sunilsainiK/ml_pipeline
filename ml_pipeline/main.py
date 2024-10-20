

import os
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.prediction import router as prediction_router

from api.training import router as training_router
from core.preprocessing import load_and_preprocess
from core.training import train_model

# Setting up logging configuration to log all information into 'app.log' under the logs directory
LOG_DIR = './ml_pipeline/logs/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(level=logging.INFO, filename=os.path.join(LOG_DIR, 'app.log'),
                    filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables for model and data paths
DATA_PATH = os.getenv("DATA_PATH", "ml_pipeline/Data/datatraining.txt")
PREPROCESSED_DATA_PATH = "./ml_pipeline/models/preprocessed_data.pkl"
MODEL_PATH = "./ml_pipeline/models/xgboost_model.pkl"

# Initialize FastAPI application
app = FastAPI()

# Include the prediction and training routers, which define the endpoints
app.include_router(prediction_router)
app.include_router(training_router)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for the lifespan of the FastAPI application.
    This function checks if the data is outdated or if new data has arrived.
    If so, it will start the preprocessing and training process.
    """
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
    """
    FastAPI startup event that runs when the application starts.
    It manages the preprocessing and training logic as part of the app lifecycle.
    """
    async with lifespan(app):
        pass  # Run the lifespan manager

@app.get("/")
async def root():
    """
    Root endpoint for the FastAPI app. Returns a welcome message.
    """
    return {"message": "Welcome to the ML Pipeline API!"}

def is_preprocessed_data_outdated(data_path, preprocessed_data_path):
    """
    Compares modification times of the raw data and preprocessed data to determine if preprocessing is required.
    Returns True if the preprocessed data is outdated or missing, False otherwise.
    """
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
    """
    Preprocess the raw data and save the preprocessed data to a file.
    """
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)
        with open(PREPROCESSED_DATA_PATH, 'wb') as f:
            import pickle
            pickle.dump((X_train, X_test, y_train, y_test), f)
        logger.info("Preprocessed data saved successfully.")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

async def start_training_process():
    """
    Start the model training process after preprocessing.
    """
    try:
        logger.info("Starting training process...")
        train_result = train_model(DATA_PATH)
        logger.info(train_result['message'])
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def check_for_new_data(data_path, preprocessed_data_path):
    """
    Checks if new data has arrived by comparing the modification times of the raw and preprocessed data.
    Returns True if new data is found, False otherwise.
    """
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
