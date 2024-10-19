

import joblib
import logging

logger = logging.getLogger(__name__)

def save_model(model, file_path: str):
    """
    Save the trained model to the specified file path.
    """
    try:
        joblib.dump(model, file_path)
        logger.info(f"Model saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise RuntimeError(f"Failed to save model: {e}")
