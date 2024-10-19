import xgboost as xgb
import json
from .preprocessing import load_and_preprocess
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from ml_pipeline.utils.save_model import save_model

PERFORMANCE_FILE_PATH = "ml_pipeline/models/model_performance.json"

def load_performance() -> float:
    """Load the current model performance from the JSON file."""
    try:
        with open(PERFORMANCE_FILE_PATH, 'r') as f:
            performance = json.load(f)
        return performance.get("roc_auc_score", 0)
    except FileNotFoundError:
        return 0.0  # If the file does not exist, return 0

def save_new_performance(roc_auc: float):
    """Save the new model performance to the JSON file."""
    with open(PERFORMANCE_FILE_PATH, 'w') as f:
        json.dump({"roc_auc_score": roc_auc}, f)

def train_model(data_file: str):
    """
    Trains an XGBoost model using RandomizedSearchCV for hyperparameter tuning.
    Compares ROC-AUC with the previous model and saves the new model if it performs better.
    """
    X_train, X_test, y_train, y_test = load_and_preprocess(data_file)

    # Train XGBoost model
    model = xgb.XGBClassifier()

    # Define hyperparameter grid for randomized search
    param_dist = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    # Random search with cross-validation
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                       n_iter=10, scoring='roc_auc', cv=3)

    # Fit the model
    random_search.fit(X_train, y_train)

    # Get the best model from randomized search
    best_model = random_search.best_estimator_

    # Make predictions on the test set
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate ROC-AUC for the new model
    new_roc_auc = roc_auc_score(y_test, y_pred_proba)
    current_roc_auc = load_performance()

    print(f"New model ROC-AUC score: {new_roc_auc}")
    print(f"Current model ROC-AUC score: {current_roc_auc}")

    # Compare with the current model performance and save if better
    if new_roc_auc > current_roc_auc:
        print("New model performs better. Saving the model...")
        save_model(best_model, "ml_pipeline/models/xgboost_model.pkl")
        save_new_performance(new_roc_auc)
        return {
            "message": "New model trained and saved successfully.",
            "new_roc_auc": new_roc_auc,
            "current_roc_auc": current_roc_auc
        }
    else:
        print("New model did not perform better. Discarding the model.")
        return {
            "message": "New model did not perform better. Training skipped.",
            "new_roc_auc": new_roc_auc,
            "current_roc_auc": current_roc_auc
        }
