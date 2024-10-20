import os
import time
from datetime import datetime
from ml_pipeline.core.preprocessing import load_and_preprocess
from ml_pipeline.core.training import train_model, load_performance, save_new_performance
from ml_pipeline.utils.save_model import save_model

# Paths
RAW_DATA_PATH = "ml_pipeline/Data/datatraining.txt"
MODEL_PATH = "ml_pipeline/models/xgboost_model.pkl"
PERFORMANCE_FILE_PATH = "ml_pipeline/models/model_performance.json"

# Time interval to check for new data (in seconds)
CHECK_INTERVAL = 60  # For example, every 60 seconds

# Step 1: Check for new data in the directory
def check_for_new_data(data_dir: str, last_checked: datetime):
    """Check if new data has been added to the directory since the last check."""
    new_data_found = False
    latest_mod_time = last_checked

    # Check all files in the directory for their modification times
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if file_mod_time > last_checked:
            new_data_found = True
            if file_mod_time > latest_mod_time:
                latest_mod_time = file_mod_time

    return new_data_found, latest_mod_time

# Step 2: Trigger model retraining pipeline
def trigger_training_pipeline():
    """Trigger the training pipeline if new data is detected."""
    print("Triggering model retraining...")

    # Train the model and evaluate performance
    new_model, new_roc_auc = train_model(RAW_DATA_PATH)

    # Load current performance
    current_roc_auc = load_performance()

    print(f"New model ROC AUC: {new_roc_auc}, Current model ROC AUC: {current_roc_auc}")

    # Ensure that current_roc_auc is a float
    current_roc_auc = float(current_roc_auc)

    # If the new model performs better, save it
    if new_model is not None and new_roc_auc > current_roc_auc:
        print("New model performs better. Saving the new model...")
        save_model(new_model, MODEL_PATH)
        # Save the new performance
        save_new_performance(new_roc_auc)
    else:
        print("New model does not perform better. Discarding the new model.")

# Step 3: Monitor for new data and trigger training when necessary
def monitor_data(data_dir: str):
    """Monitor the directory for new data and trigger training if detected."""
    last_checked = datetime.now()  # Start time of the monitoring
    while True:
        # Check for new data
        new_data_found, latest_mod_time = check_for_new_data(data_dir, last_checked)
        if new_data_found:
            print("New data detected. Starting model retraining...")
            trigger_training_pipeline()
            last_checked = latest_mod_time  # Update the last checked time
        else:
            print("No new data detected. Monitoring for changes...")

        # Wait for the next check
        time.sleep(CHECK_INTERVAL)

# Step 4: Run the monitoring process
if __name__ == "__main__":
    print("Starting data monitoring process...")
    data_directory = "ml_pipeline/Data"
    monitor_data(data_directory)
