# kmeans.py
# Demo of k-means algorithm on synthetic 2D data.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormaps
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi
from voronoi_finite_polygons_2d import *
from scipy.spatial.distance import cdist
import time

# Pause
def pause():
    plt.waitforbuttonpress()

# Update labels
# def update_labels(x, mu, s1):
#     k = len(mu) # grab the number of means
#     distances = cdist(x, mu) # cdist calculates the distance between every point in x and every point in mu
#     labels = np.argmin(distances) # finds the shortest distance
#     cmap = colormaps.get_cmap (list(range(k)))
#     s1.set_color(cmap[labels - 1,:]) # update colors
#     plt.show

#     return labels

# Update labels
def update_labels(x, mu, s1):
    k = len(mu)  # number of means
    d = cdist(x, mu)
    labels = np.argmin(d, axis=1)

    cmap = colormaps.get_cmap(name='Pastel1', lut=(list(range(k))))
    s1.set_color(cmap[labels-1, :])
    plt.show()

    return labels

# Voronoi
def voronoi(mu, v=[]):
    vor = Voronoi(mu)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    i = 0
    for region in regions:
        polygon = vertices[region]
        try:
            v[i].set_data(polygon[:, 0], polygon[:, 1])
        except:
            new_plot, = plt.plot(polygon[:, 0], polygon[:, 1], 'k-')
            v.append(new_plot)
        i += 1
    plt.show()
    pause()

# Update means
def update_means(x, mu, labels, s2):
    # Mine
    sumOfX = sum(x)
    mu = labels * sumOfX
    s2, = plt.plot(mu[:, 0], mu[:, 1], color='red', ms=10, marker='o', ls='')

    # Dr. Eicholtz
    k = len(mu)
    for i in range(k):
        mu[i,:] = np.mean(x[labels == i,:], axis=0)
    s2.set_data(mu[:,0], mu[:,1])
    plt.show()
    pause()


# Set parameters
k = 5
seed = 7
iters = 10

# Generate data
x0, y0 = make_blobs(n_samples=500, centers=[[0, 0]], cluster_std=1.0, random_state=seed)
x1, y1 = make_blobs(n_samples=300, centers=[[6, 6]], cluster_std=1.0, random_state=seed)
x2, y2 = make_blobs(n_samples=400, centers=[[-5, 5]], cluster_std=2.0, random_state=seed)
x3, y3 = make_blobs(n_samples=1000, centers=[[3, -3]], cluster_std=0.5, random_state=seed)
x4, y4 = make_blobs(n_samples=100, centers=[[-6, -4]], cluster_std=1.0, random_state=seed)

# print(x0)

x = np.concatenate((x0, x1, x2, x3, x4))

# Initialize k-means
np.random.seed(seed)
mu = np.random.uniform(-8, 8, size=(k,2))

# Show data
plt.figure(1)
plt.clf()
s1 = plt.scatter(x[:,0], x[:,1], s=6)
ax = plt.gca()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.axis('off')

s2, = plt.plot(mu[:,0], mu[:,1], color='red', ms=10, marker='o', ls='')
plt.show()

# Train data
v = []
for  i in range(iters):
    print("Iter {} of {}".format((i+1), iters))
    
    # Plot Voronoi diagram
    voronoi(mu, v)

    # Update labels
    labels = update_labels(x, mu, s1)

    # Update means
    update_means(x, mu, labels, s2)
