#import matplotlib.image as mpimg
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as sp
import pandas as pd

data_images = []

# Load in the images
for filepath in os.listdir('images/allSetsSameSize'):
    data_images.append(cv2.imread('images/allSetsSameSize/{0}'.format(filepath), 0))

print(type(data_images[0]))
print(len(data_images))
print(data_images[0].shape)

data_array = np.zeros((len(data_images), 75*75))
print(data_array.shape)
for x in range(0, len(data_images)):
    data_array[x,:] = data_images[x].flatten()
print(data_array.shape)
