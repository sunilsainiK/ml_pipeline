import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess(file_path: str, return_split=True):
    df = pd.read_csv(file_path)

    # Preprocess data
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek

    # Feature selection
    X = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'hour', 'day_of_week']]
    y = df['Occupancy']
    if not return_split:
        # Return preprocessed data (without splitting)
        return X, y
    else:
        return train_test_split(X, y, test_size=0.2, random_state=42)
