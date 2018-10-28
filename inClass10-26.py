# Import relevant libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Fixed seed
np.random.seed(3520)

# Make function f(x)
def f(x):
    return np.sin(2*np.pi*x) + np.cos(4*np.pi*x)

# Create data
x0 = np.ones((1,1000))*np.linspace(0,1,1000)
t0 = f(x0)
x0 = x0.T
t0 = t0.T

numberOfPoints = 40
x = np.ones((1,numberOfPoints)) * np.linspace(0,1,numberOfPoints)
y = f(x)
x = x.T
y = y.T

# Add noise
noise = np.random.randn(numberOfPoints,1)
x += noise*0.01
y += noise*0.2

# Get/set parameters
samples, inputs = x.shape
outputs = y.shape[1]
batchSize = 10
epochs = 1000

# Create neural network
model = Sequential()
model.add(Dense(units=300, init='normal', activation='relu', input_dim=inputs))
model.add(Dense(units=300, init='normal', activation='linear'))
model.add(Dense(units=outputs, init='normal', activation='linear'))
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')

# Train network
history = model.fit(x, y, batch_size=batchSize, epochs=epochs, verbose=2)

# Test the network
prediction = model.predict(x0)

# Plot data
plt.plot(x0, f(x0), '-', linewidth=1, color=[0,0,0])
plt.plot(x, y, 'o', linewidth=0.5, markersize=5, color=[1,0,0])
plt.plot(x0, prediction, '-', linewidth=1, color=[0,1,0])
plt.xlabel('x')
plt.ylabel('y')
plt.show(block=True)
plt.pause(5)
plt.close()
