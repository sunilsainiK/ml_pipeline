# File: ml_pipeline/utils/load_model.py
import joblib
import logging

logger = logging.getLogger(__name__)

def load_model(file_path: str):
    try:
        model = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
