import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import mglearn # library from the book:
    # Introduction to Machine Learning With Python. code in:
#https://github.com/amueller/introduction_to_ml_with_python
# Used for plot and to get datasets.

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# generate synthetic two-dimensional data
X, y = make_blobs(cluster_std=2.5, random_state=1)
# build the clustering model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print("Cluster memberships:\n{}".format(kmeans.labels_))

print(kmeans.predict(X))

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(
kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2],
markers='^', markeredgewidth=2)