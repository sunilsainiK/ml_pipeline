import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os
import pickle

# Create a logger for this module
logger = logging.getLogger(__name__)


def load_and_preprocess(file_path: str, return_split=True):
    """
    This function loads raw data from a CSV file, preprocesses it, and returns the training/testing split.

    - file_path: Path to the CSV file with raw data.
    - return_split: If True, returns the train-test split; if False, returns the full preprocessed data.
    """
    try:
        # Load the data into a pandas DataFrame
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")

        # Convert 'date' column to datetime and extract hour and day of the week
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek

        # Select features (independent variables) and target (dependent variable)
        X = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'hour', 'day_of_week']]
        y = df['Occupancy']

        if return_split:
            # Split data into training and testing sets
            return train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            return X, y

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def preprocess_input(data: dict):
    """
    Preprocesses input data for a single prediction request.

    - data: A dictionary containing the input data from the API.

    Returns:
        features_array: A numpy array of the preprocessed features.
    """
    try:
        # Convert 'date' to datetime and extract hour and day of the week
        data['date'] = pd.to_datetime(data['date'])
        data['hour'] = data['date'].hour
        data['day_of_week'] = data['date'].day_of_week

        # Create a DataFrame and select the relevant features
        df = pd.DataFrame([data])
        features = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'hour', 'day_of_week']]

        # Convert the DataFrame to a numpy array and reshape
        features_array = features.values.reshape(1, -1)
        return features_array

    except Exception as e:
        logger.error(f"Error during input preprocessing: {e}")
        raise


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_path):
    """
    Save preprocessed training and testing data to a pickle file.

    - X_train: Training feature set
    - X_test: Testing feature set
    - y_train: Training labels
    - y_test: Testing labels
    - output_path: Path to save the preprocessed data pickle file
    """
    try:
        with open(output_path, 'wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        logger.info("Preprocessed data saved successfully to: %s", output_path)
    except Exception as e:
        logger.error(f"Error saving preprocessed data: {e}")
        raise


def load_preprocessed_data(input_path):
    """
    Load preprocessed data from a pickle file.

    - input_path: Path to the pickle file.

    Returns:
        (X_train, X_test, y_train, y_test) or raises an exception if loading fails.
    """
    try:
        with open(input_path, 'rb') as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
        logger.info("Preprocessed data loaded successfully from: %s", input_path)
        return X_train, X_test, y_train, y_test
    except (EOFError, pickle.UnpicklingError) as e:
        logger.error(f"Failed to load preprocessed data. File may be corrupted or invalid: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {e}")
        raise
