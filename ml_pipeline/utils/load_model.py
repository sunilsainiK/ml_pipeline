

import joblib
import logging
import pickle
from fastapi import FastAPI, HTTPException, UploadFile, File

logger = logging.getLogger(__name__)

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