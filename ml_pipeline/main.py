# File: ml_pipeline/main.py
import os
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from ml_pipeline.api.prediction import router as prediction_router
from ml_pipeline.api.training import router as training_router
from ml_pipeline.core.preprocessing import load_and_preprocess
from ml_pipeline.utils.data_drift import detect_data_drift
from ml_pipeline.core.training import train_model

# Set up logging
LOG_DIR = './ml_pipeline/logs/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(level=logging.INFO, filename=os.path.join(LOG_DIR, 'app.log'),
                    filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = os.getenv("DATA_PATH", "ml_pipeline/Data/datatraining.txt")
PREPROCESSED_DATA_PATH = "./ml_pipeline/models/preprocessed_data.pkl"
MODEL_PATH = "./ml_pipeline/models/xgboost_model.pkl"

# Initialize FastAPI app
app = FastAPI()

# Include the prediction and training routes
app.include_router(prediction_router)
app.include_router(training_router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler to manage preprocessing, training, and monitoring.
    """
    logger.info("Starting FastAPI application...")

    if is_preprocessed_data_outdated(DATA_PATH, PREPROCESSED_DATA_PATH):
        logger.info("Preprocessed data is outdated or missing. Preprocessing...")
        await preprocess_and_save_data()

        logger.info("Training the model with updated data...")
        await start_training_process()
    else:
        logger.info("Preprocessed data is up-to-date. Checking for new data.")
        if check_for_new_data(DATA_PATH, PREPROCESSED_DATA_PATH):
            logger.info("New data detected. Reprocessing and retraining the model...")
            await preprocess_and_save_data()
            await start_training_process()

    yield


@app.on_event("startup")
async def startup_event():
    """
    Startup event to initialize preprocessing and training.
    """
    async with lifespan(app):
        pass


@app.get("/")
async def root():
    return {"message": "Welcome to the ML Pipeline API!"}


def is_preprocessed_data_outdated(data_path, preprocessed_data_path):
    """
    Check if preprocessed data is outdated by comparing the modification times.
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
    Preprocess the raw data and save the preprocessed data.
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
    Start the model training process.
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
    Check if new data has arrived by comparing modification times.
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
