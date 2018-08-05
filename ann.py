# Import all necessary modules
import tensorflow
import keras

from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset, from 'https://www.kaggle.com/CooperUnion/cardataset'
dataset = pd.read_csv('data.csv')

# Create the data and the labels correseponding to the data
X = dataset.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]].values
y = dataset.iloc[:, 15].values

# Preprocess the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing.imputation import Imputer

label_encoder_make = LabelEncoder()
X[:, 0] = label_encoder_make.fit_transform(X[:, 0])

label_encoder_engineType = LabelEncoder()
X[:, 2] = label_encoder_engineType.fit_transform(X[:, 2].astype(str))

label_encoder_transmission = LabelEncoder()
X[:, 5] = label_encoder_transmission.fit_transform(X[:, 5])

label_encoder_wheels = LabelEncoder()
X[:, 6] = label_encoder_wheels.fit_transform(X[:, 6].astype(str))

label_encoder_size = LabelEncoder()
X[:, 8] = label_encoder_size.fit_transform(X[:, 8])

label_encoder_style = LabelEncoder()
X[:, 9] = label_encoder_style.fit_transform(X[:, 9])

imp = Imputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

# one-hot encode
onehotencoder_make = OneHotEncoder(categorical_features=[0])
X = onehotencoder_make.fit_transform(X).toarray()
X = X[:, 1:]

onehotencoder_engine = OneHotEncoder(categorical_features=[48])
X = onehotencoder_engine.fit_transform(X).toarray()
X = X[:, 1:]

onehotencoder_transmission = OneHotEncoder(categorical_features=[60])
X = onehotencoder_transmission.fit_transform(X).toarray()
X = X[:, 1:]

onehotencoder_wheels = OneHotEncoder(categorical_features=[64])
X = onehotencoder_wheels.fit_transform(X).toarray()
X = X[:, 1:]

onehotencoder_size = OneHotEncoder(categorical_features=[69])
X = onehotencoder_size.fit_transform(X).toarray()
X = X[:, 1:]

onehotencoder_style = OneHotEncoder(categorical_features=[83])
X = onehotencoder_style.fit_transform(X).toarray()
X = X[:, 1:]
