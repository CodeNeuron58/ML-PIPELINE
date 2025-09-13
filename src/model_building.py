import numpy as np 
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle
import yaml

# custom logger
from logger import get_logger
logger = get_logger("model_building")


def load_data(train_data_path):
    logger.info(f"Loading training data from {train_data_path}")
    try:
        train_data = pd.read_csv(train_data_path)
        logger.info(f"Training data loaded successfully with shape {train_data.shape}")
        return train_data
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise


def split_data(train_data):
    logger.info("Splitting training data into features and labels...")
    try:
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        logger.info(f"Split completed. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        return X_train, y_train
    except Exception as e:
        logger.error(f"Error splitting training data: {e}")
        raise


def model_building(X_train, y_train, params_file="params.yaml"):
    logger.info("Building GradientBoostingClassifier model...")
    try:
        params = yaml.safe_load(open(params_file))["model_building"]

        clf = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"]
        )

        logger.info(f"Training model with parameters: n_estimators={params['n_estimators']}, "
                    f"learning_rate={params['learning_rate']}")
        clf.fit(X_train, y_train)
        logger.info("Model training completed successfully.")
        return clf
    except Exception as e:
        logger.error(f"Error in model building: {e}")
        raise


def save_model(clf, model_path="model.pkl"):
    logger.info(f"Saving trained model to {model_path}")
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(clf, file)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def main():
    logger.info("Starting model building pipeline...")
    train_data = load_data('data/features/train.csv')
    X_train, y_train = split_data(train_data)
    clf = model_building(X_train, y_train)
    save_model(clf, model_path="model.pkl")
    logger.info("Model building pipeline completed successfully.")


if __name__ == '__main__':
    main()