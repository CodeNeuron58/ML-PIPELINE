import sklearn.feature_extraction.text as text
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np 

import os

def load_data(train_data,test_data):
    train_data = pd.read_csv('data/processed/train.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    return train_data,test_data

def handle_nan(X_train, X_test):
    # Fill missing values with empty strings or a placeholder
    X_train = pd.Series(X_train).fillna("missing_text").values
    X_test = pd.Series(X_test).fillna("missing_text").values
    return X_train, X_test

# splitting x and y
def split_data(train_data,test_data):
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values

    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values
    return X_train,y_train,X_test,y_test

# Apply bag of words 
def apply_bag_of_words(X_train,X_test):
    vectorizer = CountVectorizer(max_features=500)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return X_train_vectorized,X_test_vectorized
    

def save_data(train_df,test_df,data_path):
    os.makedirs(data_path, exist_ok=True)
    train_df.to_csv(os.path.join(data_path, 'train.csv'),index = False)
    test_df.to_csv(os.path.join(data_path, 'test.csv'), index = False)
    
def combining_data(X_train_vectorized,X_test_vectorized,y_train,y_test,data_path):
    train_df = pd.DataFrame(X_train_vectorized.toarray())
    train_df["labels"] = y_train
    test_df = pd.DataFrame(X_test_vectorized.toarray())
    test_df["labels"] = y_test
    save_data(train_df,test_df,data_path)

def featured_data(train_data,test_data,data_path):
    train_data,test_data = load_data(train_data,test_data)
    X_train,y_train,X_test,y_test = split_data(train_data,test_data)
    X_train, X_test = handle_nan(X_train, X_test)
    X_train_vectorized,X_test_vectorized = apply_bag_of_words(X_train,X_test)
    combining_data(X_train_vectorized,X_test_vectorized,y_train,y_test,data_path)

if __name__ == '__main__':
    data_path = os.path.join('data', 'features')
    featured_data(train_data="data/processed/train.csv",test_data="data/processed/test.csv",data_path=data_path)
    