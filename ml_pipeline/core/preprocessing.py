# File: ml_pipeline/core/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def load_and_preprocess(file_path: str, return_split=True):
    try:
        df = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek

        X = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'hour', 'day_of_week']]
        y = df['Occupancy']

        if return_split:
            return train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            return X, y
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise
