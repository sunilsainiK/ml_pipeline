
import pandas as pd
from sklearn.model_selection import train_test_split  # Function to split data into training and testing sets
import logging

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
