import numpy as np 
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import pickle
import json

# custom logger
from logger import get_logger
logger = get_logger("model_evaluation")

def load_data(test_data):
    logger.info(f"Loading test data from {test_data}")
    try:
        test_data = pd.read_csv('data/features/test.csv')
        logger.info(f"Test data loaded successfully with shape {test_data.shape}")
        return test_data
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise


def split_data(test_data):
    logger.info("Splitting features and target...")
    try:
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        logger.info(f"Split completed. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        return X_test, y_test
    except Exception as e:
        logger.error(f"Error splitting test data: {e}")
        raise


def load_model(model_path):
    logger.info(f"Loading model from {model_path}")
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model...")
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise


def save_metrics(metrics, output_path='metrics.json'):
    logger.info(f"Saving metrics to {output_path}")
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info("Metrics saved successfully.")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise


def main():
    logger.info("Starting model evaluation pipeline...")
    test_data = load_data('data/features/test.csv')
    X_test, y_test = split_data(test_data)
    model = load_model('model.pkl')
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics)
    logger.info("Model evaluation pipeline completed successfully.")


if __name__ == '__main__':
    main()