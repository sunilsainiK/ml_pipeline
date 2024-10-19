# File: ml_pipeline/utils/save_model.py
import joblib
import logging

logger = logging.getLogger(__name__)

def save_model(model, file_path: str):
    try:
        joblib.dump(model, file_path)
        logger.info(f"Model saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise RuntimeError(f"Failed to save model: {e}")
