# File: ml_pipeline/monitoring.py
import os
import time
import requests
import logging
from ml_pipeline.utils.data_drift import detect_data_drift
from ml_pipeline.core.preprocessing import load_and_preprocess

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Monitoring configuration
DATA_DIRECTORY = "ml_pipeline/Data/"
PREPROCESSED_DATA_PATH = "ml_pipeline/models/preprocessed_data.pkl"
MODEL_PATH = "ml_pipeline/models/xgboost_model.pkl"
CHECK_INTERVAL = 240  # Check every 240 seconds

def check_for_new_data(data_directory: str, last_checked_time: float) -> bool:
    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        if os.path.isfile(filepath):
            file_mod_time = os.path.getmtime(filepath)
            if file_mod_time > last_checked_time:
                return True
    return False

def trigger_training_pipeline():
    logger.info("Triggering training pipeline due to new data or detected drift.")
    try:
        # Assuming we have an API route to trigger training via FastAPI
        response = requests.post("http://localhost:8000/train")
        logger.info(f"Training pipeline triggered. Status: {response.status_code}")
    except Exception as e:
        logger.error(f"Error triggering training pipeline: {e}")

def monitor_data():
    """
    Monitor the data directory and detect data drift or new data for retraining.
    """
    last_checked_time = time.time()

    while True:
        time.sleep(CHECK_INTERVAL)  # Sleep for defined interval
        logger.info("Checking for new data or drift...")

        if check_for_new_data(DATA_DIRECTORY, last_checked_time):
            logger.info("New data detected. Triggering retraining...")
            trigger_training_pipeline()
            last_checked_time = time.time()

        else:
            # Check for data drift
            try:
                X_train, _, _, _ = load_and_preprocess(os.path.join(DATA_DIRECTORY, "datatraining.txt"))
                with open(PREPROCESSED_DATA_PATH, 'rb') as f:
                    import pickle
                    X_train_old, _, _, _ = pickle.load(f)

                drift_results = detect_data_drift(X_train, X_train_old)
                if any(result["drift_detected"] for result in drift_results.values()):
                    logger.warning("Data drift detected!")
                    trigger_training_pipeline()
                else:
                    logger.info("No data drift detected.")
            except Exception as e:
                logger.error(f"Error during drift detection: {e}")

if __name__ == "__main__":
    monitor_data()
