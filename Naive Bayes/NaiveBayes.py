# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 20:22:07 2020

@author: Willie Man

Goals: Trying to predict heart disease diagnosis

Research/Resources:
    
    
Dataset: https://www.kaggle.com/ronitf/heart-disease-uci and https://archive.ics.uci.edu/ml/datasets/heart+disease
    
Future Endeavors:
    
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Import Data:
data = pd.read_csv("heart.csv")

# Pre-Processing and data cleaning:

# Feature Selection:
X = data.iloc[:, np.arange(13)].values
y = data['target']



# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split the data:
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Build the Model
clf = GaussianNB()
clf = clf.fit(X_train, y_train)

# Predictions:
predictions = clf.predict(X_test)


# Evaluate
clf.score(X_train, y_train)
clf.score(X_test, y_test)
roc_score = roc_auc_score(y_test, predictions)

"""
First run are in the mid to high, 80s. Not bad!

"""



""" Scoping: """

# Load libraries
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create Gaussian Naive Bayes object
classifer = GaussianNB()

# Train model
model = classifer.fit(features, target)

model.predict([[5.9,3,5.1,1.8]])
