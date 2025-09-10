import numpy as np 
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import pickle
import json

def load_data(test_data):
    test_data = pd.read_csv('data/features/test.csv')
    return test_data
def split_data(test_data):
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    return X_test,y_test
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }


def save_metrics(metrics, output_path='metrics.json'):
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

        
if __name__ == '__main__':
    test_data = load_data('data/features/test.csv')
    X_test,y_test = split_data(test_data)
    model = load_model('model.pkl')
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics)
    print("✅ Evaluation complete. Metrics saved to 'metrics.json'.")