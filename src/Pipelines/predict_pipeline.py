import os 
import sys
import pickle

import numpy as np
import pandas as pd

import dataclasses
from utilities import load_object, convert_softmax
from preprocessor import cleaner, one_hot_encoder
from exception import CustomException
from logger import logging

class PredictPipeline():
    def predict(self,sentence):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            encoder_path = os.path.join('artifacts','encoder.pkl')
            model = load_object(file_path=model_path)
            encoder_object = load_object(file_path=encoder_path)
            data = [cleaner(sentence)]
            data = one_hot_encoder(data,encoder_object=encoder_object)
            result = model.predict(data)
            result = convert_softmax(result)
            result = result[0]
            prediction = ""
            if result == 0:
                prediction = "Sadness"
            elif result == 1:
                prediction = "Joy"
            elif result== 2:
                prediction = "Love"
            elif result== 3:
                prediction = "Anger"
            elif result== 4:
                prediction = "Fear"
            else:
                prediction = "Surprise"
            
            return prediction

        except Exception as e:
            raise CustomException(e,sys)

    