import os 
import sys
from exception import CustomException
from logger import logging


import pandas as pd

from dataclasses import dataclass
import preprocessor

class DataTransformation:
    def initiate_data_transformation(self,train_path,test_path,raw_data_path):
        
        try:
            logging.info("Data Transformation has been initiated")

            train_data= pd.read_csv(train_path)
            test_data= pd.read_csv(test_path)
            raw_data = pd.read_csv(raw_data_path)
            logging.info("Preprocessing train and test data")

            train_data = preprocessor.preprocess(train_data)
            test_data = preprocessor.preprocess(test_data)
            raw_data_path = preprocessor.preprocess(raw_data)

            logging.info("Preprocessing is complete")
            
            return (
                train_data,
                test_data,
                raw_data
            )
        
        except Exception as e:
            raise CustomException(e,sys)
