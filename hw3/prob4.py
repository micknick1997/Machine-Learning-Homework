# Import libraries
# import importlib
from . import mnist
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Variables
def sigmoid(x):
    return 1 / (1.0 + np.exp(-x))

learningRate = 0.001
epochs = 500
np.random.seed(3520)

# mnist = input('mnist.py')
# importlib.import_module(mnist)

# Load data
trainingImages = mnist.load_images("problem4Dataset/train-images-idx3-ubyte")
testingImages = mnist.load_images("problem4Dataset/t10k-images-idx3-ubyte")
trainingLabels = mnist.load_labels("problem4Dataset/train-labels-idx1-ubyte")
testingLabels = mnist.load_labels("problem4Dataset/t10k-labels-idx1-ubyte")

# Create neural network
model = Sequential()
model.add(Dense(units=784, init='normal', activation='sigmoid', input_dim=trainingImages.shape))
model.add(Dense(units=300, init='normal', activation='sigmoid'))
model.add(Dense(units=10, init='normal', activation='sigmoid'))
model.summary()

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=learningRate))

# Train neural network
training = model.fit(trainingImages, trainingLabels, batch_size=100, epochs=epochs, verbose=2)

# Test neural network

# Make plot