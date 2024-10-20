import os
import time
import requests
import logging
import pickle
from utils.data_drift import detect_data_drift
from core.preprocessing import load_and_preprocess, save_preprocessed_data

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Monitoring configuration
DATA_DIRECTORY = "Data/"
PREPROCESSED_DATA_PATH = "models/preprocessed_data.pkl"
MODEL_PATH = "models/xgboost_model.pkl"
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


def regenerate_pickle_file():
    """
    Regenerate the preprocessed pickle file if it is invalid or missing.
    """
    logger.info("Regenerating the preprocessed pickle file.")
    try:
        # Load the raw data and preprocess it
        X_train, X_test, y_train, y_test = load_and_preprocess(os.path.join(DATA_DIRECTORY, "datatraining.txt"))
        save_preprocessed_data(X_train, X_test, y_train, y_test, PREPROCESSED_DATA_PATH)  # Save the new pickle file
        logger.info("Preprocessed data file regenerated successfully.")
    except Exception as e:
        logger.error(f"Error regenerating pickle file: {e}")
        raise


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
                # Load the preprocessed data from the pickle file
                with open(PREPROCESSED_DATA_PATH, 'rb') as f:
                    X_train_old, X_test_old, y_train_old, y_test_old = pickle.load(f)
                logger.info("Preprocessed data loaded successfully.")

                # Load the current data for drift detection
                X_train, X_test, y_train, y_test = load_and_preprocess(os.path.join(DATA_DIRECTORY, "datatraining.txt"))

                # Detect data drift
                drift_results = detect_data_drift(X_train, X_train_old)
                if any(result["drift_detected"] for result in drift_results.values()):
                    logger.warning("Data drift detected!")
                    trigger_training_pipeline()
                else:
                    logger.info("No data drift detected.")

            except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
                # Handle cases where the pickle file is corrupted or missing
                logger.error(f"Error during drift detection or loading preprocessed data: {e}")

                # Regenerate the pickle file if there's an error
                regenerate_pickle_file()
            except Exception as e:
                # Catch-all for any other exceptions
                logger.error(f"Unexpected error during drift detection: {e}")
                regenerate_pickle_file()  # Regenerate pickle file on any other error


if __name__ == "__main__":
    monitor_data()
