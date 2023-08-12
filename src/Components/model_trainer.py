import os
import sys

import numpy as np
import pandas as pd

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping

import data_transformation

from dataclasses import dataclass
from exception import CustomException
from logger import logging
from utilities import save_object, model_builder, get_corpus, convert_softmax
from encoder import OneHotEncoder


@dataclass
class Model_training_config:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    encoder_path = os.path.join('artifacts','encoder.pkl')
    

class ModelTrainer:
    def __init__(self):
        self.model_training_config=Model_training_config()

    def initiate_model_training(self,train_data,test_data,raw_data):
        try:
            encoder_object = OneHotEncoder()
            train_x = np.array(get_corpus(train_data,encoder_object=encoder_object))
            train_y = np.array(pd.get_dummies(train_data.Sentiment))
            test_x = np.array(get_corpus(test_data, encoder_object=encoder_object))
            test_y = np.array(test_data.Sentiment)
            
            raw_data_x = np.array(get_corpus(raw_data,encoder_object=encoder_object))
            raw_data_y = np.array(pd.get_dummies(raw_data.Sentiment))

            logging.info("The train and test data have been split into x and y")

            save_object(
                obj=encoder_object,
                file_path=self.model_training_config.encoder_path
            )
            logging.info('Encoder saved as a pickle file')
            
            logging.info("Evaluating best model and parameters")
            
            top_params= {}
            best_accuracy= 0

            layers = [[32],[64],[128],[256]]
            activations = ['relu','sigmoid']
            callback = EarlyStopping(monitor='loss', patience=10)
            param_grid= dict(
                callbacks=[callback],
                batch_size=[128,256],
                epochs = [200]
            )

            for layer in layers:
                for activation in activations:
                    model = KerasClassifier(build_fn= model_builder,activation = activation,layers= layer)
                    grid = GridSearchCV(estimator = model,param_grid=param_grid,n_jobs=4)
                    grid_result = grid.fit(train_x,train_y)
                    dic = grid_result.best_params_
                    best_model = model_builder(layers=layer,activation=activation)
                    best_model.fit(train_x,train_y,callbacks=callback,batch_size = dic['batch_size'],epochs = 200)
                    pred_y = convert_softmax(best_model.predict(test_x))
                    accuracy = f1_score(test_y,pred_y,average = 'micro')
                    if accuracy>best_accuracy:
                        best_accuracy = accuracy
                        dic.update({'layers':layer,'activation':activation})
                        top_params=dic
            logging.info("Parameters have been tested and accuracy has been recorded")

            logging.info("Best Model has an accuracy of {}".format(best_accuracy))
            top_model = model_builder(layers=top_params['layers'],activation=top_params['activation'])
            top_model.fit(raw_data_x,raw_data_y,callbacks=[callback],batch_size = top_params['batch_size'],epochs=200)

            save_object(
                file_path=self.model_training_config.trained_model_file_path,
                obj=top_model
            )

            logging.info("Top Model saved as pickle file")
            return self.model_training_config.trained_model_file_path


        except Exception as e:
            raise CustomException(e,sys)
