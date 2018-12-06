import json
import numpy as np
import objectpath as op
from skimage.io import imread

set1 = imread("set1.jpg")
set2 = imread("set2.jpg")
set3 = imread("set3.jpg")
set4 = imread("set4.jpg")

print(set1)

with open("handwritingTrainingData.txt") as datafile:
    datafile = datafile.read()
    data = json.loads(str(datafile))

np_data = np.asarray(data)
#print(np_data)
