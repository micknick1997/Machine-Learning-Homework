# Import libraries
# import importlib
#from . import mnist
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from skimage.util import montage

np.random.seed(3520)

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

# mnist = input('mnist.py')
# importlib.import_module(mnist)

# Load data
trainingImages = load_images("problem4Dataset/train-images-idx3-ubyte")
testingImages = load_images("problem4Dataset/t10k-images-idx3-ubyte")
trainingLabels = load_labels("problem4Dataset/train-labels-idx1-ubyte")
testingLabels = load_labels("problem4Dataset/t10k-labels-idx1-ubyte")

# Reshape and normalize data
# print(trainingImages.shape)
# print(len(trainingLabels))
trainingImages = trainingImages.reshape(60000, 28, 28)
testingImages = testingImages.reshape(10000, 28, 28)
trainingImages = trainingImages.astype('float32')
testingImages = testingImages.astype('float32')
trainingImages /= 255.0
testingImages /= 255.0
trainingLabels = keras.utils.to_categorical(trainingLabels, num_classes=10)
testingLabels = keras.utils.to_categorical(testingLabels, num_classes=10)

print(trainingLabels.shape)

# Create neural network
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(units=784, activation='sigmoid'))
model.add(Dense(units=300, activation='sigmoid'))
model.add(Dense(units=10, activation='sigmoid'))
model.summary()

model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=keras.optimizers.Adam(lr=learningRate))

# Train neural network
training = model.fit(trainingImages, trainingLabels, epochs=epochs, verbose=2, batch_size=100)

# Test neural network
score = model.evaluate(testingImages, testingLabels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

prediction = model.predict(testingImages)
prediction = np.argmax(prediction, axis=1)
# testingLabels, names = pd.factorize(testingLabels)
# c = confusion_matrix(testingLabels, prediction)
# print("Confusion matrix")
# print(c)

# Make plot
plt.plot(score[1], '-')
plt.plot(testingImages, prediction, '-', color=[0,1,0])
plt.xlabel('x')
plt.ylabel('y')
plt.show(block=True)
plt.pause(5)
plt.close()