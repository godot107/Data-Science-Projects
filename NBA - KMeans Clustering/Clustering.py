# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:06:01 2020

@author: Willie Man

Research/Resources:
   
    
Getting Started:     https://realpython.com/k-means-clustering-python/
Reference Project: https://www.kaggle.com/vjchoudhary7/kmeans-clustering-in-customer-segmentation
Graphic: https://stackoverflow.com/questions/26139423/plot-different-color-for-different-categorical-levels-using-matplotlib

K-means documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
Data Source: https://archive.ics.uci.edu/ml/datasets/Automobile

https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
https://analyticsindiamag.com/beginners-guide-to-k-means-clustering/
Looking into clustering in more dimensions.: https://towardsdatascience.com/how-to-cluster-in-high-dimensions-4ef693bacc6

"""

# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

# Scoping:

from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4) # Builds the modle
kmeans.fit(X) 
y_kmeans = kmeans.predict(X) # Assign the coordinates to a cluster, according to n_clusters

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_ # coordinates for the center of the cluster.
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

# My Attempt:

# Import data:
data = pd.read_csv('imports-85.csv')

# Feature selection: 
X = data.iloc[:, [10,11]].values # length, width

# Deciding the right amount of clusters:
wcss=[] # WCSS is the sum of squares of the distances of each data point in all clusters to their respective centroids.
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting and evaluating the elbow plot
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show() # looking for the elbow, the optimal amount of N_ cluster is 5


# Building the Model:
kmeans = KMeans(n_clusters= 5, init='k-means++', random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X) # assign cluster to the coordinate
predictions = kmeans.labels_ # also, assign cluster to each coordinate
centers = kmeans.cluster_centers_ # coordinates of the centroids


# Plot / Explore:

#Exploratory (no cluster)
colors = {'convertible':'red', 'hardtop':'blue', 'hatchback':'green', 'sedan':'black', 'wagon':'orange'}
plt.scatter(data['length'], data['width'], c=data['body-style'].apply(lambda x: colors[x]))
plt.xlabel("Length")
plt.ylabel("Width")
plt.show()

# The assigned Clusters
plt.scatter(X[:, 0], X[:,1], c=y_kmeans) # X[:, 0] means X data, all the rows, and column 0
plt.xlabel("Length")
plt.ylabel("Width")
plt.show() # insights aren't useful at least with width and length.




