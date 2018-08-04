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

label_encoder_make = LabelEncoder()
X[:, 0] = label_encoder_make.fit_transform(X[:, 0])

label_encoder_engineType = LabelEncoder()
X[:, 2] = label_encoder_engineType.fit_transform(X[:, 2].astype(str))

label_encoder_transmission = LabelEncoder()
X[:, 5] = label_encoder_transmission.fit_transform(X[:, 5])

label_encoder_wheels = LabelEncoder()
X[:, 6] = label_encoder_wheels.fit_transform(X[:, 6].astype(str))

label_encoder_doors = LabelEncoder()
X[:, 8] = label_encoder_doors.fit_transform(X[:, 8])

label_encoder_style = LabelEncoder()
X[:, 9] = label_encoder_style.fit_transform(X[:, 9])

# binary encode
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 0:]

