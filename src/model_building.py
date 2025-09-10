import numpy as np 
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle

def load_data(train_data_path):
    train_data = pd.read_csv(train_data_path)
    return train_data

def split_data(train_data):
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    return X_train,y_train

def model_building(X_train,y_train):
    clf = GradientBoostingClassifier(n_estimators=50)
    clf.fit(X_train,y_train)
    return clf

def save_model(clf):
    pickle.dump(clf,open('model.pkl','wb'))

if __name__ == '__main__':
    train_data = load_data('data/features/train.csv')
    X_train,y_train = split_data(train_data)
    clf = model_building(X_train,y_train)
    save_model(clf)
