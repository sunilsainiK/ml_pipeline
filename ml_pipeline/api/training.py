from fastapi import APIRouter, HTTPException
from ml_pipeline.core.training import train_model
import os
import logging

# Create an API router for training endpoints
router = APIRouter()

# Create a logger for this module
logger = logging.getLogger(__name__)


@router.post("/train")
async def train():
    """
    This endpoint triggers the model training process.
    It reads data, trains the model, and saves it if the new model is better.
    """
    try:
        # Get the data file path from environment variable or default
        data_path = os.getenv("DATA_PATH", "ml_pipeline/Data/datatraining.txt")
        logger.info(f"Starting training with data from {data_path}...")

        # Call the train_model function from the core module
        result = await train_model(data_path)  # If train_model is async, otherwise just call train_model()

        logger.info("Model trained and saved successfully.")
        return {
            "message": "Model trained and saved successfully",
            "details": result
        }

    except FileNotFoundError as fnf_error:
        logger.error(f"Data file not found: {fnf_error}")
        raise HTTPException(status_code=404, detail=f"Data file not found: {fnf_error}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")
