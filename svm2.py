# svm2.py
# Example support vector machine (SVM).

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap

# Create synthetic data
n = 90 # number of samples
classes = 5 # number of classes
seed = 6 # for repeatability

cmap_light = ListedColormap(['#FFAFAF','#AFAFFF','#F6D587','#90EE90', '#EE82EE'])
cmap_bold = ListedColormap(['red','blue','orange','green','purple'])

x, t = make_blobs(n, centers=classes, random_state=seed)

# Train SVM
clf = svm.SVC(kernel='linear',
    C=10.0,
    decision_function_shape='ovr')
clf.fit(x,t)

# Compute decision boundary
plt.figure(1)
plt.clf()
plt.scatter(x[:,0], x[:,1], c=t, s=30, cmap=cmap_bold)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 300)
yy = np.linspace(ylim[0], ylim[1], 300)
XX, YY = np.meshgrid(xx, yy)
z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
z = z.reshape(XX.shape)


# Show results
plt.pcolormesh(xx, yy, z, cmap=cmap_light, alpha=1.0)
plt.scatter(x[:,0], x[:,1], c=t, s=30,
    cmap=cmap_bold,
    edgecolors='k')
plt.show()

