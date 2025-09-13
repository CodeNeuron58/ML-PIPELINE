from nltk.inference.tableau import testTableauProver
import pandas as pd
import numpy as np 
import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer,WordNetLemmatizer

from logger import get_logger

logger = get_logger("data_preprocessing")

def load_data(train_data, test_data):
    logger.info("Loading train and test data...")
    try:
        train_data = pd.read_csv('data/raw/train.csv')
        test_data = pd.read_csv('data/raw/test.csv')
        logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error loading train/test data: {e}")
        raise


nltk.download("wordnet")
nltk.download("stopwords")

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def removing_stopwords(text):
    stop_words = stopwords.words("english")
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    text = "".join([i for i in text if not i.isdigit()])
    return text

def removing_punctuation(text):
    text = text.split()
    text = [word for word in text if word not in string.punctuation]
    return " ".join(text)

def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_urls(text):
    text = re.sub(r"http\S+", "", text)
    return text

def remove_small_sentences(text):
    text = text.split()
    text = [word for word in text if len(word) > 3]
    return " ".join(text)

def final_preprocessing(text):
    try:
        text = lemmatization(text)
        text = removing_stopwords(text)
        text = removing_numbers(text)
        text = removing_punctuation(text)
        text = lower_case(text)
        text = removing_urls(text)
        text = remove_small_sentences(text)
        return text

    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        raise
    
    

def save_data(train_data, test_data, data_path):
    logger.info(f"Saving processed data to {data_path}")
    try:
        os.makedirs(data_path, exist_ok=True)
        train_file = os.path.join(data_path, 'train.csv')
        test_file = os.path.join(data_path, 'test.csv')
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        logger.info(f"Processed train and test data saved successfully at {data_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise

    
def main():
    data_path = os.path.join('data', 'processed')
    logger.info("Starting preprocessing pipeline...")
    train_data,test_data = load_data(train_data="data/raw/train.csv",
                                     test_data="data/raw/test.csv")
    logger.info("Preprocessing train and test data...")
    
    train_data['content'] = train_data['content'].apply(final_preprocessing)
    test_data['content'] = test_data['content'].apply(final_preprocessing)
    logger.info("Preprocessing completed...")
    
    save_data(train_data, test_data, data_path)
    logger.info("Preprocessing pipeline completed...")


if __name__ == '__main__':
    main()



