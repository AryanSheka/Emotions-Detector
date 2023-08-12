# Emotion Detector

This project takes in a sentence from a user and gives the emotion represented by the sentence, one of 6 emotions namely sadness, joy, anger, love, fear and surprise.

Uses an embedding layer and LSTM recurrent neural network to predict the emotions with an f1_score of 82.7% with average set to micro in test dataset.

Uses one hot encoding to vectorize the data.

Data ingestion is done from a MYSQL database in local system.

User interface is of the form of a web application developed using Flask Web application framework.

Docker is used for Containerization.

## Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.txt
```
pip install -r 'requirements.txt'
```

## Loading data
Data loading is done from a MYSQL database in the local system in the [data ingestion](https://github.com/AryanSheka/Financial-Sentiment-Analysis/blob/744c3182201ac3606a5189a35567d6d009c0a8a9/src/Components/data_ingestion.py) file and the train, test and raw data are stored in the [artifacts](https://github.com/AryanSheka/Financial-Sentiment-Analysis/tree/744c3182201ac3606a5189a35567d6d009c0a8a9/artifacts) folder in the form of a csv file.

The original dataset is obtained from [dair.ai](https://huggingface.co/datasets/dair-ai/emotion/viewer/split/train?row=48)

## Data Transformation
Data cleaning and transformation is done in the [data transformation](https://github.com/AryanSheka/Financial-Sentiment-Analysis/blob/744c3182201ac3606a5189a35567d6d009c0a8a9/src/Components/data_transformation.py) file.

Encoding is done using one hot encoding and the encoder is saved as a pickle file in [artifacts](https://github.com/AryanSheka/Financial-Sentiment-Analysis/tree/744c3182201ac3606a5189a35567d6d009c0a8a9/artifacts)

## Model training
Model training and hyperparameter tuning are done in the [model_trainer](https://github.com/AryanSheka/Financial-Sentiment-Analysis/blob/744c3182201ac3606a5189a35567d6d009c0a8a9/src/Components/model_trainer.py) file.

Hyperparameter tuning is done using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and model with best accuracy with test data is trained with the entire dataset and saved as a pickle file in [artifacts](https://github.com/AryanSheka/Financial-Sentiment-Analysis/tree/744c3182201ac3606a5189a35567d6d009c0a8a9/artifacts)

## Running the web app
The web app can be run by running the app.py file 
```
python app.py
```

The [app](https://github.com/AryanSheka/Financial-Sentiment-Analysis/blob/744c3182201ac3606a5189a35567d6d009c0a8a9/app.py) file contains all the flask code used to run the web app which loads web page which is made using html stored in the [templates](https://github.com/AryanSheka/Financial-Sentiment-Analysis/tree/744c3182201ac3606a5189a35567d6d009c0a8a9/templates) file.


## Demo


https://github.com/AryanSheka/Emotions-Detector/assets/77437286/4c032999-2d22-4504-8013-107b0387ac22


## License

[MIT](https://choosealicense.com/licenses/mit/)
