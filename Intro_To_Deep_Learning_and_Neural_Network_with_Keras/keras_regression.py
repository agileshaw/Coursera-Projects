from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import numpy as np

def regression_model(n_cols):
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

if __name__ == '__main__':
    dataset = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
    print(dataset.shape)
    print(dataset.head())
    print(dataset.describe())

    columns = dataset.columns
    print(columns)

    features = dataset[columns[columns != 'Strength']]
    target = dataset['Strength']
    print(features.head())
    print(target)

    features_norm = (features - features.mean())/features.std()
    print(features_norm.head())
    n_cols = features.shape[1]
    print(n_cols)

    model = regression_model(n_cols)
    model.fit(features_norm, target, validation_split=0.3, epochs=100, verbose=2)
