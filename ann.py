# Import all necessary modules
import tensorflow as tf
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

labelencode_array = [0, 2, 5, 6, 8, 9]

for i in labelencode_array:
    labelencoder = LabelEncoder()
    X[:, i] = labelencoder.fit_transform(X[:, i].astype(str))


imp = Imputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

# one-hot encode

onehotencode_array = [0, 48, 60, 64, 69, 83]

for i in onehotencode_array:
    onehotencoder_make = OneHotEncoder(categorical_features=[i])
    X = onehotencoder_make.fit_transform(X).toarray()
    X = X[:, 1:]
    
# Split the dataset into the Training set and Test set
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu',))
model.add(Dense(1))

optimizer = tf.train.RMSPropOptimizer(0.001)

model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

# Display training progress by printing a single dot for each completed epoch.
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 100 == 0: print('')
    print((epoch), end='\n')

EPOCHS = 10000

# Store training stats
history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

model_answer = model.predict(X_test)
print("\n")

correct = 0
incorrect = 0

for i in range(1, len(model_answer)):
    lower_bound = 0.9 * y_test[i]
    upper_bound = 1.15 * y_test[i]
    
    if ((model_answer[i] > lower_bound) and (model_answer[i] < upper_bound)):
        correct += 1
    else:
        incorrect += 1
        
print("The model got", correct, "right, and", incorrect, "wrong")
