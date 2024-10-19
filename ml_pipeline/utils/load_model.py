import joblib

def load_model(file_path: str):
    """Load the saved XGBoost model from a file."""
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
