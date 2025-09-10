from nltk.inference.tableau import testTableauProver
import pandas as pd
import numpy as np 
import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer,WordNetLemmatizer


def load_data(train_data,test_data):
    train_data = pd.read_csv('data/raw/train.csv')
    test_data = pd.read_csv('data/raw/test.csv')
    return train_data,test_data

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
    text = lemmatization(text)
    text = removing_stopwords(text)
    text = removing_numbers(text)
    text = removing_punctuation(text)
    text = lower_case(text)
    text = removing_urls(text)
    text = remove_small_sentences(text)
    
    return text


def save_data(train_data, test_data, data_path):
    os.makedirs(data_path, exist_ok=True)
    train_data.to_csv(os.path.join(data_path, 'train.csv'))
    test_data.to_csv(os.path.join(data_path, 'test.csv'))

if __name__ == '__main__':
    data_path = os.path.join('data', 'processed')
    train_data,test_data = load_data(train_data="data/raw/train.csv",test_data="data/raw/test.csv")
    
    train_data['content'] = train_data['content'].apply(final_preprocessing)
    test_data['content'] = test_data['content'].apply(final_preprocessing)
    save_data(train_data, test_data, data_path)



