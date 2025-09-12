# Import required libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

def load_data(data_url):
    df = pd.read_csv(data_url)
    return df

def preprocess(df):
    df.drop(columns=['tweet_id'], inplace=True)
    final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
    final_df['sentiment'] = final_df['sentiment'].map({'happiness': 1, 'sadness': 0})
    return final_df


def split_data(df, params_file="params.yaml"):
    params = yaml.safe_load(open(params_file))["data_ingestion"]["test_size"]
    train_data,test_data = train_test_split(df, test_size=params, random_state=42)
    return train_data,test_data

def save_data(train_data, test_data, data_path):
    os.makedirs(data_path, exist_ok=True)
    train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)

def run_data_ingestion(data_url, data_path):
    df = load_data(data_url)
    df = preprocess(df)
    train_data, test_data = split_data(df)
    save_data(train_data, test_data, data_path)

def main():
    data_url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
    data_path = os.path.join('data', 'raw')
    run_data_ingestion(data_url, data_path)
    
if __name__ == '__main__':
    main()




