import os
import pickle
import json
from sklearn.metrics import roc_auc_score
from ml_pipeline.core.preprocessing import load_and_preprocess, save_preprocessed_data
from ml_pipeline.utils.save_model import save_model
from ml_pipeline.core.training import train_model as xgb_train_model

# Paths to use in this script
RAW_DATA_PATH = "ml_pipeline/Data/datatraining.txt"  # Update this path to your raw data
PREPROCESSED_DATA_PATH = "ml_pipeline/models/preprocessed_data.pkl"
MODEL_PATH = "ml_pipeline/models/xgboost_model.pkl"
PERFORMANCE_FILE_PATH = "ml_pipeline/models/model_performance.json"


# Step 1: Create necessary directories if they don't exist
def create_directories():
    directories = ["ml_pipeline/models"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")


# Step 2: Preprocess the data and save the preprocessed data
def preprocess_data():
    # Preprocess the data from the raw data file and split into training and testing sets
    X_train, X_test, y_train, y_test = load_and_preprocess(RAW_DATA_PATH)

    # Save the preprocessed data to the specified pickle file
    save_preprocessed_data(X_train, X_test, y_train, y_test, PREPROCESSED_DATA_PATH)
    print(f"Preprocessed data saved to {PREPROCESSED_DATA_PATH}")

    return X_train, X_test, y_train, y_test


# Step 3: Load performance metrics from a previous model (if exists)
def load_performance():
    if os.path.exists(PERFORMANCE_FILE_PATH):
        with open(PERFORMANCE_FILE_PATH, 'r') as f:
            performance_data = json.load(f)
        return performance_data.get("roc_auc_score", 0)
    return 0  # Return 0 if no previous performance file exists


# Step 4: Save new performance metrics if the model performs better
def save_new_performance(roc_auc):
    performance_data = {"roc_auc_score": roc_auc}
    with open(PERFORMANCE_FILE_PATH, 'w') as f:
        json.dump(performance_data, f)
    print(f"Updated performance saved to {PERFORMANCE_FILE_PATH}")


# Step 5: Train the model and update performance if better than the previous model
def train_and_save_model():
    print("Starting the training process...")

    # Train the model using the XGBoost training function with hyperparameter tuning
    result = xgb_train_model(RAW_DATA_PATH)

    # Check if the new model performs better and save it
    if "new_roc_auc" in result:
        new_roc_auc = result["new_roc_auc"]
        print(f"New ROC AUC score: {new_roc_auc}")
    else:
        print(result["message"])

    return result


# Step 6: Run all steps for training, performance comparison, and saving
if __name__ == "__main__":
    print("Starting first-time setup...")

    # Create necessary directories
    create_directories()

    # Preprocess the data and save the preprocessed files
    preprocess_data()

    # Train the model, calculate roc_auc_score, compare to the current model, and save the new one if it performs better
    result = train_and_save_model()

    if "new_roc_auc" in result:
        print(f"Training completed. New model ROC AUC: {result['new_roc_auc']}")
    else:
        print(result["message"])

    print("First-time setup completed.")

