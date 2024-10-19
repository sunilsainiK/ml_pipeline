import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json


def calculate_baseline_metrics(X: pd.DataFrame, baseline_path: str):
    """Calculate and save baseline metrics for each feature."""
    baseline_metrics = {
        "means": X.mean().to_dict(),
        "stds": X.std().to_dict(),
        "mins": X.min().to_dict(),
        "maxs": X.max().to_dict(),
    }

    # Save baseline metrics to a JSON file
    baseline_file_path = f"{baseline_path}/baseline_metrics.json"
    with open(baseline_file_path, 'w') as f:
        json.dump(baseline_metrics, f)

    return baseline_metrics


def detect_data_drift(new_data: pd.DataFrame, baseline_metrics: dict) -> dict:
    """Detects data drift by comparing new data to baseline metrics."""
    drift_results = {}

    for column in new_data.columns:
        if column in baseline_metrics["means"]:  # Numerical features
            ks_statistic, p_value = ks_2samp(new_data[column], baseline_metrics["means"][column])
            drift_results[column] = {
                "ks_statistic": ks_statistic,
                "p_value": p_value,
                "drift_detected": p_value < 0.05  # Common threshold for significance
            }
        else:
            # Handle categorical features (could also include additional logic here)
            drift_results[column] = {
                "drift_detected": not np.all(
                    new_data[column].value_counts(normalize=True) == baseline_metrics["means"].get(column, {})),
            }

    return drift_results
