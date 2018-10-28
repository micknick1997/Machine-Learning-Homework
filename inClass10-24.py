# Import relevant libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense

# Fix seed for reproducability
np.random.seed(3520)

# Load the dataset
data = pd.read_csv("iris.csv", header=None)
dataSet = data.values
x = dataSet[:,:4].astype('float32')
y = dataSet[:,4]
encoder = LabelBinarizer()
t = encoder.fit_transform(y)
# Get/set relevant parameters
samples, inputs = x.shape
outputs = len(np.unique(y))
batchSize = 50
epochs = 1000
learningRate = 0.01

# Create neural network
model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=inputs))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=outputs, activation='softmax'))
model.summary()

sdg = keras.optimizers.SGD(lr=learningRate)
model.compile(loss='mean_squared_error', optimizer=sdg, metrics=['accuracy'])

# Train neural network
history = model.fit(x, t, batch_size=batchSize, epochs=epochs, verbose=2)

# Test neural network
score = model.evaluate(x, t, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

prediction = model.predict(x)
prediction = np.argmax(prediction, axis=1)
y, names = pd.factorize(y)
c = confusion_matrix(y, prediction)
print("Confusion matrix")
print(c)