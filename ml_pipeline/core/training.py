# File: ml_pipeline/core/training.py
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from ml_pipeline.core.preprocessing import load_and_preprocess
from ml_pipeline.utils.save_model import save_model
from ml_pipeline.utils.load_model import load_model
import logging
import os
import json

logger = logging.getLogger(__name__)
PERFORMANCE_FILE_PATH = "ml_pipeline/models/model_performance.json"

def load_performance():
    try:
        with open(PERFORMANCE_FILE_PATH, 'r') as f:
            performance = json.load(f)
        return performance.get("roc_auc_score", 0)
    except FileNotFoundError:
        logger.warning("Performance file not found. Returning default ROC-AUC: 0.")
        return 0.0

def save_new_performance(roc_auc):
    with open(PERFORMANCE_FILE_PATH, 'w') as f:
        json.dump({"roc_auc_score": roc_auc}, f)

def train_model(data_file):
    try:
        logger.info("Starting model training...")
        X_train, X_test, y_train, y_test = load_and_preprocess(data_file)
        model = xgb.XGBClassifier()
        param_dist = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }

        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, scoring='roc_auc', cv=3)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        new_roc_auc = roc_auc_score(y_test, y_pred_proba)
        current_roc_auc = load_performance()

        if new_roc_auc > current_roc_auc:
            logger.info(f"New model ROC-AUC ({new_roc_auc}) outperforms current ROC-AUC ({current_roc_auc}).")
            save_model(best_model, os.getenv("MODEL_PATH", "ml_pipeline/models/xgboost_model.pkl"))
            save_new_performance(new_roc_auc)
            return {"message": "New model trained and saved successfully", "new_roc_auc": new_roc_auc}
        else:
            logger.info("New model does not outperform the current model.")
            return {"message": "New model did not outperform the current model", "new_roc_auc": new_roc_auc}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
