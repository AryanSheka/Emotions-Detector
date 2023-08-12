import pandas as pd 
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

def cleaner(text, stem=False):
    stop_words = stopwords.words('english')
    stemmer = PorterStemmer()
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9$]+"
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


def one_hot_encoder(s,encoder_object):
    vocab_size = 10000
    rep = ([encoder_object.encode(word,vocab_size) for word in s])
    doc = pad_sequences(rep,maxlen = 30,padding = 'pre')
    return doc


def preprocess(data):
    dd = data
    dd.Sentence = dd.Sentence.apply(lambda x : cleaner(x))
    return dd
