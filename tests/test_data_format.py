import os
import pandas as pd
import pytest

DATA_PATH = "../ml_pipeline/Data/datatraining.txt"

def test_data_existence():
    """Test that the data file exists."""
    assert os.path.exists(DATA_PATH), "Training data file does not exist."

def test_data_format():
    """Test that the training data has the correct format."""
    # Load the data
    df = pd.read_csv(DATA_PATH)

    # Check for expected columns
    expected_columns = ["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"

    # Check for expected data types
    assert df['date'].dtype == 'object', "Date column should be of type 'object'."
    assert df['Temperature'].dtype == 'float64', "Temperature column should be of type 'float64'."
    assert df['Humidity'].dtype == 'float64', "Humidity column should be of type 'float64'."
    assert df['Light'].dtype == 'float64', "Light column should be of type 'float64'."
    assert df['CO2'].dtype == 'float64', "CO2 column should be of type 'float64'."
    assert df['HumidityRatio'].dtype == 'float64', "HumidityRatio column should be of type 'float64'."

    # Check for any missing values
    assert df.isnull().sum().sum() == 0, "Data contains missing values."
