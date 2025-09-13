# Import required libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

from logger import get_logger

logger = get_logger("data_ingestion")

def load_data(data_url):
    logger.info("Loading data from url")
    try:
        df = pd.read_csv(data_url)
        logger.info("Loading data from url")
        return df
    except Exception as e:
        logger.error(f"Error loading data from url: {e}")
        raise


def preprocess(df):
    logger.info("Preprocessing data")
    try:
        df.dropna(inplace=True)
        logger.info("Preprocessing data")
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'] = final_df['sentiment'].map({'happiness': 1, 'sadness': 0})
        logger.info("Preprocessing data Completed")
        return final_df
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise



def split_data(df, params_file="params.yaml"):
    logger.info("Split data")
    try:
        params = yaml.safe_load(open(params_file))["data_ingestion"]["test_size"]
        train_data,test_data = train_test_split(df, test_size=params, random_state=42)
        logger.info("Spliting data Completed")
        return train_data,test_data
    except Exception as e :
        logger.error(f"Error Spliting data: {e}")
        raise


def save_data(train_data, test_data, data_path):
    
    os.makedirs(data_path, exist_ok=True)
    train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)

def run_data_ingestion(data_url, data_path):
    try:
        df = load_data(data_url)
        df = preprocess(df)
        train_data, test_data = split_data(df)
        save_data(train_data, test_data, data_path)
    except Exception as e:
        logger.error(f"Error running data ingestion: {e}")
        raise

def main():
    data_url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
    data_path = os.path.join('data', 'raw')
    run_data_ingestion(data_url, data_path)
    
if __name__ == '__main__':
    main()




