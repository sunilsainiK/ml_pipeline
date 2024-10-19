
from fastapi import APIRouter, HTTPException
from ml_pipeline.core.training import train_model
import os
import logging

# Create an API router for training endpoints
router = APIRouter()

# Create a logger for this module
logger = logging.getLogger(__name__)


@router.post("/train")
def train():
    """
    This endpoint triggers the model training process.
    It reads data, trains the model, and saves it if the new model is better.
    """
    try:
        # Get the data file path from environment variable or default
        data_path = os.getenv("DATA_PATH", "ml_pipeline/Data/datatraining.txt")

        # Call the train_model function from the core module
        result = train_model(data_path)

        return {"message": "Model trained and saved successfully", "details": result}

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
