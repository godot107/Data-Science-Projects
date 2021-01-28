# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 21:41:23 2020

@author: Willie Man


Research/Resources:

"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Import the data:

# Feature Selection

# Build the Model

# Evaluate the Model


""" REFERENCE """
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(max_depth = 4, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train))) # pure leaves and therefore some overfitting
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
feature_names=cancer.feature_names, impurity=False, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)  # doesn't seem to want to be rendered for whatever reason.... most likely versioning.
