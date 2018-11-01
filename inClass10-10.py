import os
import sys
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import graphviz
import matplotlib.pyplot as plt

# Define activation function(s)


def sigmoid(x):
    return 1 / (1.0 + np.exp(-x))


def sigmoidPrime(s):
    return s * (1 - s)


def linear(x):
    return x 

# Define training function(s)


def feedForward(X, weight1, weight2):
    bias = np.ones((X.shape[0], 1))
    a1 = np.dot(np.concatenate((bias, X), axis=1), weight1)
    z = sigmoid(a1)
    a2 = np.dot(np.concatenate((bias, z), axis=1), weight2)
    y = linear(a2)

    return y, z


def backPropogate(X, weight1, weight2, target, index, eta):
    # First, extract some sample

    xSample = X[index, :]
    xSample.shape = (1, 2)
    targetSample = target[index]

    # Feedforward to get neuron activations

    activation, z = feedForward(xSample, weight1, weight2)
    activation.shape = (2, 1)
    z.shape = (2, 1)

    # Update hidden-to-output neurons

    weight_jk = weight2 - eta * (activation - targetSample) * np.vstack([1, z])

    # Update input-to-hidden neurons

    sensitivity = -(activation - targetSample) * weight2[1:] * z * (1 - z)
    weight_ij = weight1 + eta * sensitivity.transpose() * \
        np.vstack([1, xSample])

    # Return the updated weights

    return weight_ij, weight_jk

# Initialize Data


x1Data = [0, 1, -1, -2, 3, 1, -2, -3]
x2Data = [0, 2, 2, -2, 0, -3, 3, -3]
# X = np.array([[0, 0],
#               [1, 2],
#               [-1, 2],
#               [-2, -2],
#               [3, 0],
#               [1, -3],
#               [-2, 3],
#               [-3, -3]])
X = np.array([[]])
targetData = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
# -1 specifies unknown dimension. Python does the rest
targetData.shape = (-1, 1)
eta = 0.2  # learning rate
weight1 = np.array([[0, 0],
                    [1, 0],
                    [0, 1]])
weight2 = np.array([0, 1, 1])
weight2.shape = (-1, 1)

# Feed-forward data and compute accuracy

pred, Z = feedForward(X, weight1, weight2)
yData = np.zeros(pred.shape)
yData[pred < 0] = -1
yData[pred >= 0] = 1
accuracy = float((sum(targetData == yData)/len(targetData))) * 100.0

print('iter 0 --> accuracy = {:0.1f}%'.format(accuracy))

# Back-propogate the error

totalIterations = 1000
accuracy = np.empty((totalIterations, 1))
for i in range(totalIterations):
    # Backpropogate the error
    ind = i % X.shape
    weight1, weight2 = backPropogate(X, weight1, weight2, targetData, ind, eta)

    # Recompute accuracy
    pred, Z = feedForward(X, weight1, weight2)
    yData = np.zeros(pred.shape)
    yData[pred < 0] = -1
    yData[pred >= 0] = 1
    accuracy[i] = float(sum(targetData == yData) / len(targetData)) * 100
    print('iter {} --> accuracy = {:0.1f}%'.format(i + 1, float(accuracy[i])))
plt.plot(accuracy)
plt.show
