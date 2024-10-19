# File: ml_pipeline/api/training.py
from fastapi import APIRouter, HTTPException
from ml_pipeline.core.training import train_model
import os
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/train")
def train():
    try:
        data_path = os.getenv("DATA_PATH", "ml_pipeline/Data/datatraining.txt")
        result = train_model(data_path)
        return {"message": "Model trained and saved successfully", "details": result}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
