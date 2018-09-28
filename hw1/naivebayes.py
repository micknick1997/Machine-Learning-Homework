import numpy as np
from collections import Counter
import sklearn as skl
import pandas as pd

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

testFile = pd.read_table('data/trainingdata.txt',
                          sep=' ',
                          header=None,
                          names=['docNum', 'wordNum', 'timesRecorded'])

testFile['docNum'] = testFile.label.map({'alt.atheism': 1, 'comp.graphics': 2,
                                         'com.os.ms-windows.misc': 3, 'comp.sys.ibm.pc.hardware': 4,
                                         'comp.sys.mac.hardware': 5, 'comp.windows.x': 6,
                                         'misc.forsale': 7, 'rec.autos': 8,
                                         'rec.motorcycles': 9, 'rec.sport.baseball': 10,
                                         'rec.sport.hockey': 11, 'sci.crypt': 12,
                                         'sci.electronics': 13, 'sci.med': 14,
                                         'sci.space': 15, 'soc.religion.christian': 16,
                                         'talk.politics.guns': 17, 'talk.politics.mideast': 18,
                                         'talk.politics.misc': 19, 'talk.religion.misc': 20})
