import os
import pickle
import logging
from fastapi import FastAPI, HTTPException
from ml_pipeline.core.training import train_model
from ml_pipeline.utils.load_model import load_model
from ml_pipeline.utils.save_model import save_model
from contextlib import asynccontextmanager
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
from ml_pipeline.core.preprocessing import load_and_preprocess  # Import the function from preprocessing.py

# Set up logging
logging.basicConfig(level=logging.INFO)  # You can adjust the logging level here
logger = logging.getLogger(__name__)

# Define the FastAPI app
app = FastAPI()

# Define the paths for data and model files
DATA_PATH = "ml_pipeline/Data/datatraining.txt"
MODEL_PATH = "./ml_pipeline/models/xgboost_model.pkl"
PREPROCESSED_DATA_PATH = "./ml_pipeline/models/preprocessed_data.pkl"  # Path for saving the preprocessed data
PROCESSED_DATA_PATH = "./ml_pipeline/models/processed_data.pkl"  # Path for saving the split data

class PredictionInput(BaseModel):
    Temperature: float
    Humidity: float
    Light: float
    CO2: float
    HumidityRatio: float
    hour: int
    day_of_week: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler to manage startup and shutdown logic.
    Preprocesses and saves data before splitting, then splits the data afterward.
    """
    logger.info("Starting the FastAPI application...")

    # Step 1: Check if the data file exists
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}. Please ensure the data file is available.")
        yield
        return

    # Step 2: Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        logger.info(f"Model file not found at {MODEL_PATH}. Training the model...")

        # Preprocess the data if it hasn't been done already
        if not os.path.exists(PREPROCESSED_DATA_PATH):
            logger.info(f"Preprocessing the data from {DATA_PATH}...")

            # Preprocess without splitting
            X, y = load_and_preprocess(DATA_PATH, return_split=False)

            # Save the preprocessed data before splitting
            with open(PREPROCESSED_DATA_PATH, "wb") as f:
                pickle.dump((X, y), f)
            logger.info(f"Preprocessed data saved at {PREPROCESSED_DATA_PATH}.")
        else:
            logger.info(f"Preprocessed data already exists at {PREPROCESSED_DATA_PATH}.")

        # Now split the data into train and test sets
        if not os.path.exists(PROCESSED_DATA_PATH):
            logger.info("Splitting the data into train and test sets...")

            # Load preprocessed data
            with open(PREPROCESSED_DATA_PATH, "rb") as f:
                X, y = pickle.load(f)

            # Split the data
            X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH, return_split=True)

            # Save the split data
            save_model([X_train, X_test, y_train, y_test], PROCESSED_DATA_PATH)
            logger.info(f"Processed train/test data saved at {PROCESSED_DATA_PATH}.")

        # Train the model
        message = train_model(DATA_PATH)  # Train your model on the split data
        logger.info(message['message'])  # Log the training result message

    else:
        logger.info(f"Model file found at {MODEL_PATH}. Loading the model...")
        # Attempt to load the model to ensure it's valid
        try:
            model = load_model(MODEL_PATH)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}. Re-training the model...")
            message = train_model(DATA_PATH)
            logger.info(message['message'])

    yield


# Attach the lifespan handler to the FastAPI app
app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "ML Data Pipeline with FastAPI - Initial setup complete"}


@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Endpoint to run a prediction using the trained model.
    It accepts input data and returns the prediction result.
    """
    global model

    if model is None:
        raise HTTPException(status_code=400, detail="Model is not loaded or trained.")

    # Prepare the input for the model
    input_df = pd.DataFrame([input_data.dict()])  # Convert input data to DataFrame for XGBoost

    # Run prediction
    try:
        prediction = model.predict(xgb.DMatrix(input_df))
        predicted_label = int(prediction[0])  # Assuming binary classification
        return {"prediction": predicted_label}
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")


@app.get("/status")
def get_status():
    """
    Check the status of the data and model files.
    """
    data_exists = os.path.exists(DATA_PATH)
    model_exists = os.path.exists(MODEL_PATH)

    return {
        "data_exists": data_exists,
        "model_exists": model_exists,
        "message": "System ready" if data_exists and model_exists else "System not fully set up"
    }

