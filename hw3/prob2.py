# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense

# Activation function
def activationFunction(x):
    alpha = 1.716
    beta = float((2/3))
    return alpha * np.tanh(beta * x)

# Create grid
x = np.arange(-5, 5.2, 0.2)
xy = np.meshgrid(x, x)

# Feed forward

# I struggled with this part of the assignemnt. 
# Can I come in and work through it with you?
