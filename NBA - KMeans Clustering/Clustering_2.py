# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:35:34 2020

@author: Willie Man


Research/Resources: https://www.basketball-reference.com/leagues/NBA_2020.html

Data Source: https://www.basketball-reference.com/leagues/NBA_2020.html

How to make a correlation plot: https://datatofish.com/correlation-matrix-pandas/
How to adjust canvas: https://stackoverflow.com/questions/41519991/how-to-make-seaborn-heatmap-larger-normal-size

Future Endeavors: 
    
1. Would like to add some interactivity where I can hover over to the graph.


"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()  # for plot styling
import numpy as np


# Import the data:
data = pd.read_csv("Team Per Game Stats.csv")


# Feature Analysis
corr = data.corr()
plt.figure(figsize=(16, 20))
sns.heatmap(corr, annot = True) # Basically calculates R2 in a matrix.
plt.show() #Skimming at the plot, TRB (total rebound) and DRB (defensive rebound) is strongly correlated, which makes sense..  FG and PTS are strgonly correlated, which makes sense. 2P% and DRB show some correlation.  
scoping = corr > .7 # if only I can pull the relationships out, but I can't.


# Feature selection: 
#X = data.iloc[:, [20,21]].values # STL and BLK

X = data.iloc[:, [17,12]].values # DRB, 2P% 


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
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)



# Graphing Steals vs Blocks:
plt.scatter(data['STL'], data['BLK'])
plt.xlabel("Average Steals")
plt.ylabel("Average Blocks")
plt.title("Team Defense Analysis: Steals vs Blocks")
plt.show()

#plt.scatter(X[:, 0], X[:,1], c=y_kmeans.apply(lambda x: colors[x])) # X[:, 0] means X data, all the rows, and column 0// I tried to change the coloring.
plt.scatter(X[:, 0], X[:,1], c=y_kmeans) 
plt.xlabel("Average Steals")
plt.ylabel("Average Blocks")
plt.show() # some insights

# Graphing 2P% and DRB
plt.scatter( data['DRB'], data['2P%'])
plt.xlabel("DRB")
plt.ylabel("2P%")
plt.show()

plt.scatter(X[:, 0], X[:,1], c=y_kmeans) 
plt.xlabel("Average DRB")
plt.ylabel("Average 2P%")
plt.show() # Insights are OK.  This roughly shows the different tiers of defensive rebounding ranks.  This shows that defensive creates offense.  Defensive rebounds could create fast break points for a score. and FBPs 




