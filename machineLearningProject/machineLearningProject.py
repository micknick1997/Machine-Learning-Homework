#import matplotlib.image as mpimg
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as sp
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten

training_data = []
testing_data = []

# Load in the images
for filepath in os.listdir('images/trainingData'):
    training_data.append(cv2.imread('images/trainingData/{0}'.format(filepath), 0))

for filepath in os.listdir('images/testingData'):
    testing_data.append(cv2.imread('images/testingData/{0}'.format(filepath), 0))

# Put images into numpy array
training_array = np.zeros((len(training_data), 75*75))
for x in range(0, len(training_data)):
    training_array[x,:] = training_data[x].flatten()
print("Training images shape: ", training_array.shape)

testing_array = np.zeros((len(testing_data), 75*75))
for x in range(0, len(testing_data)):
    testing_array[x, :] = testing_data[x].flatten()
print("Testing images shape: ", testing_array.shape)

# Read labels from csv file
training_labels = pd.read_csv('trainingLabels.csv', header=None)
training_labels = training_labels.values
x_train_labels = training_labels[:,:26].astype('int')
print("Training labels shape: ", x_train_labels.shape)

testing_labels = pd.read_csv('testingLabels.csv', header=None)
testing_labels = testing_labels.values
x_test_labels = testing_labels[:,:26].astype('int')
print("Testing labels shape: ", x_test_labels.shape)

# Make labels work a "one or none" encoding 
training_lb = preprocessing.LabelBinarizer()
training_lb = training_lb.fit_transform(x_train_labels.flatten())
print("Training labels shape after binarization: ", training_lb.shape)

testing_lb = preprocessing.LabelBinarizer()
testing_lb = testing_lb.fit_transform(x_test_labels.flatten())
print("Testing labels shape after binarization: ", testing_lb.shape)

# Set relevent parameters
batch_size = 312
epochs = 100
learning_rate = 0.01

# Neural Network
model = Sequential()
model.add(Dense(units=5625, init='normal', activation='relu', input_dim=5625))
model.add(Dense(units=300, init='normal', activation='sigmoid'))
model.add(Dense(units=300, activation='sigmoid'))
model.add(Dense(units=26, activation='softmax'))
model.summary()

model.compile(loss='mean_squared_error', metrics=[
              'accuracy'], optimizer=keras.optimizers.Adam(lr=learning_rate))
training = model.fit(training_array, training_lb, epochs=epochs, verbose=2, batch_size=batch_size)

score = model.evaluate(testing_array, testing_lb, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

prediction = model.predict(testing_array)


