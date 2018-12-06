import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

data_x, data_y, data_labels = np.loadtxt("data.txt", unpack=True)
color = ['red' if l == 0 else 'green' for l in data_labels]
print(data_y)
plt.scatter(data_x, data_y, color=color)
plt.show()

data_xy = np.stack((data_x, data_y), axis=-1)
print(data_xy)

linked = linkage(data_xy, method='single')
dn = dendrogram(linked)
plt.show()