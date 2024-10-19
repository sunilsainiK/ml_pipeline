import joblib

def save_model(model, file_path: str):
    """Save the trained model to a file."""
    try:
        joblib.dump(model, file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {str(e)}")
