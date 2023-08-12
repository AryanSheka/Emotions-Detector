import os 
import sys
import dill
import pickle

import numpy as np
import pandas as pd
import mysql.connector as connection
from numpy import argmax

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation

from preprocessor import one_hot_encoder
from exception import CustomException
from logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def get_dataframe():
    try:
        mydb = connection.connect(host="localhost", database = 'emotions',user="root", passwd="12345678",use_pure=True)
        query = "Select * from sentiment;"
        result_dataFrame = pd.read_sql(query,mydb)
        mydb.close()
        return result_dataFrame
    
    except Exception as e:
        mydb.close()
        raise CustomException(e,sys)
    

def model_builder(layers,activation):
    model = Sequential()
    for i,nodes in enumerate(layers):
        nodes=int(nodes)
        if i == 0:
            model.add(Embedding(10000,40,input_length = 30))
            model.add(LSTM(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(LSTM(nodes))
            model.add(Dropout(0.3))
        
    model.add(Dense(6,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_corpus(data, encoder_object):
    corpus=[]
    for i in data.Sentence:
        corpus.append(i)
    corpus = one_hot_encoder(corpus,encoder_object)
    return corpus

def convert_softmax(pred):
    y_pred=[]
    for i in pred:
        y_pred.append(argmax(i))
    y_pred = np.array(y_pred)
    return y_pred
