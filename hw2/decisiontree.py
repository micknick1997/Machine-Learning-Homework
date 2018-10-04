# decisiontree.py
"""Predict Parkinson's disease based on dysphonia measurements using a decision tree."""

import os
import sys
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import graphviz
import scipy as sp

# Relevant directories and filenames
root = '~/Documents/Machine Learning Homework/hw2/'  # edit as needed
#
# feel free to define other useful filename here
#

# Load data from relevant files

attributes = np.loadtxt(os.path.expanduser(root) + "data/attributes.txt", str)
trainingData = np.loadtxt(os.path.expanduser(
    root) + "data/trainingdata.txt", delimiter=',')
trainingLables = np.loadtxt(
    os.path.expanduser(root) + "data/traininglabels.txt")
testingData = np.loadtxt(os.path.expanduser(
    root) + "data/testingdata.txt", delimiter=',')
testingLabels = np.loadtxt(os.path.expanduser(root) + "data/testinglabels.txt")


# print(trainingData.shape)
# print(len(trainingLables))
# print(len(attributes))

# Train a decision tree via information gain on the training data

decisionTree = DecisionTreeClassifier(criterion="entropy")
decisionTree.fit(trainingData, trainingLables)

# Test the decision tree

dTTest = decisionTree.predict(testingData)


# Compute the confusion matrix on test data

# print(dTTest)
# print(testingLabels)
print(confusion_matrix(dTTest, testingLabels))

# Visualize the tree using graphviz
trainingTree = decisionTree.fit(trainingData, trainingLables)
tree.export_graphviz(trainingTree, out_file="trainingTree.dot")
testingTree = decisionTree.fit(testingData, testingLabels)
tree.export_graphviz(testingTree, out_file="testingTree.dot")
