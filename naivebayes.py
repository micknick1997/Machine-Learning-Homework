import numpy as np
from collections import Counter

def probOfY(y):
    totalCount = 0
    specificCount = 0
    
    labelFile = open("data/traininglabels.txt", "r")
    allLines = labelFile.readlines()
    totalCount = len(allLines)
    for line in allLines:
        for char in line.split():
            if char == str(y):
                specificCount += 1
    probability = specificCount/totalCount
    labelFile.close()
    return probability

def probOfXGivenY(y):
    columnOne = 0
    columnTwo = 1
    columnThree = 2
    columnData = []
    trainingData = []
    delimeter = " "
    lookup=str(y)
    trainingLabels = open("data/traininglabels.txt", "r")
    for num, line in enumerate(trainingLabels, 1):
        if lookup in line:
            trainingData.append(lookup.split(delimeter)[0])
    # with open("data/traininglabels.txt") as trainingLabels:
    #     trainingLabels.read().split("\n")[lineNumbers]
    # index = [x for x in range(len(labels)) if str(y) in labels[x].lower()]
    trainingData = open("data/trainingdata.txt", "r")
    lines = trainingData.readlines()
    # Search column 1
    for x in lines:
        columnData.append(x.split(delimeter)[columnOne])
    
    trainingData.close()
    #trainingLabels.close()

    print(trainingData)

probOfXGivenY(2)