# Import libraries
# import importlib
#from . import mnist
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from skimage.util import montage

# Variables/functions
def sigmoid(x):
    return 1 / (1.0 + np.exp(-x))

# mnist.py
def load_images(filename):
    with open(filename, 'rb') as fid:
        # Read magic number
        magic = np.fromfile(fid, '>i4', 1)
        assert magic[0] == 2051, "Bad magic number in {} (expected 2051, but got {})".format(
            filename, magic[0])

        # Read number and size of images
        num_images = np.fromfile(fid, '>i4', 1)
        num_rows = np.fromfile(fid, '>i4', 1)
        num_cols = np.fromfile(fid, '>i4', 1)

        # Read image data
        images = np.fromfile(fid, '>u1').reshape(
            (num_images[0], num_rows[0], num_cols[0])).transpose((1, 2, 0))
        return images


def load_labels(filename):
    with open(filename, 'rb') as fid:
        # Read magic number
        magic = np.fromfile(fid, '>i4', 1)
        assert magic[0] == 2049, "Bad magic number in {} (expected 2049, but got {})".format(
            filename, magic[0])

        # Read number and size of images
        num_images = np.fromfile(fid, '>i4', 1)

        # Read image data
        labels = np.fromfile(fid, '>u1').reshape((num_images[0], -1))
        return labels


def show_images(images, N=1, shape=None):
    # Show N random samples from the dataset.
    ind = np.random.choice(images.shape[2], N, replace=False)
    ind.shape = (len(ind),)

    if shape is None:
        s = int(np.ceil(N**(0.5)))
        shape = (s, s)
    m = montage(images[:, :, ind].transpose(2, 0, 1), grid_shape=shape)
    plt.imshow(m, cmap='gray')
    plt.axis('off')
    plt.show()

learningRate = 0.001
epochs = 500
np.random.seed(3520)

# mnist = input('mnist.py')
# importlib.import_module(mnist)

# Load data
trainingImages = load_images("problem4Dataset/train-images-idx3-ubyte")
testingImages = load_images("problem4Dataset/t10k-images-idx3-ubyte")
trainingLabels = load_labels("problem4Dataset/train-labels-idx1-ubyte")
testingLabels = load_labels("problem4Dataset/t10k-labels-idx1-ubyte")

# Reshape and normalize data
trainingImages = trainingImages.reshape(60000, 784)
testingImages = testingImages.reshape(10000, 784)
trainingImages = trainingImages.astype('float32')
testingImages = testingImages.astype('float32')
trainingImages /= 255
testingImages /= 255

# Create neural network
model = Sequential()
model.add(Dense(units=784, init='normal', activation='sigmoid', input_dim=784))
model.add(Dense(units=300, init='normal', activation='sigmoid'))
model.add(Dense(units=10, init='normal', activation='sigmoid'))
model.summary()

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=learningRate))

# Train neural network
training = model.fit(trainingImages, trainingLabels, batch_size=128, epochs=epochs, verbose=2, validation_data=(testingImages, testingLabels))

# Test neural network

# Make plot
