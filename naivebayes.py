import numpy as np

def main():
    vocabList = []
    docNum = 0
    vocabFile = open("data/vocabulary.txt", "r")

    vocab = vocabFile.readlines()
    for word in vocab:
        vocabList.append(word)
    #print(len(vocabList))

    trainingDocsFile = open("data/traininglabels.txt", "r")
    trainingDocs = trainingDocsFile.readlines()
    for line in trainingDocs:
        docNum += 1
    #print(docNum)



main()