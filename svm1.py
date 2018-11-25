# svm1.py
# Example support vector machine (SVM).

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# Create synthetic data
n = 40 # number of samples
classes = 2 # number of classes
seed = 6 # for repeatability
x, t = make_blobs(n, centers=classes, random_state=seed)

# Train SVM
clf = svm.SVC(kernel='linear', C=5)
clf.fit(x,t)

# Show data
plt.scatter(x[:,0], x[:,1], c=t, s=30, cmap=plt.cm.get_cmap("Paired"))
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Compute decision boundary
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
XX, YY = np.meshgrid(xx, yy)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
z = clf.decision_function(xy).reshape(XX.shape)

# Show decision boundary
ax.contour(XX, YY, z, 
    colors='k',
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],
    s=100,
    linewidth=1,
    facecolors='none',
    edgecolors='k') 
plt.show()

