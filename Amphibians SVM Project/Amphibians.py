# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:49:06 2020

@author: Willie Man

Goals:  Attempt to guess if a frong is a common toad based on provided features.


Research/Resources:

Link to dataset:  https://archive.ics.uci.edu/ml/datasets/Amphibians

Feature selection: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
Feature selection sklearn lib: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

Doing some research as to why my program is taking a long time to load: https://stackoverflow.com/questions/17455302/gridsearchcv-extremely-slow-on-small-dataset-in-scikit-learn/23813876

Update:
Best ROC is 60%... which is barely above guessing half and half. I tried parameter tuning and feature selection, but no luck.

"""

# Import our libraries
from sklearn import datasets
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import RFE
import seaborn as sns
import matplotlib.pyplot as plt



# Import data:
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00528/dataset.csv', sep = ";", skiprows= 1)

# X and y
#X = data.iloc[:, [3,4,5,6,7,8,9,11,12,13,14,15]] # from Feature Selection Stage, the top 5 important columns are 4,6,9,14,15
X = data.iloc[:, [4,6,9,14,15]]
y = data['Common toad']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 



# Build our SVM classifier:
clf = SVC()

# Feature Selection:
"""
selector = RFE(clf, n_features_to_select=5, step=1)
selector = selector.fit(X,y)
selector.support_


# This helps see which features are highly correlated with each other.  
cor = data.iloc[:, [3,4,5,6,7,8,9,11,12,13,14,15] ].corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds) # how I can make it more readable?
plt.show()
"""


# Grid Search (optional)
param_grid = {'kernel':['poly'],'C':[100]} #, 'gamma':[.1,1,10]} # it was determined that poly for kernel, 100 for C are optimal hyper parameters, but it takes a while for Gamma to process for some reason.
clf = GridSearchCV(clf, param_grid)


# Train the model

clf.fit(X_train, y_train)

# Best combination of parameters to use based on the gridsearch
clf.best_params_


# Predictions:
y_pred = clf.predict(X_test)

# Evaluation:
print(classification_report(y_test,y_pred))

# 1st attempt: 2 True Negative, 18 False Positive, 2 False Negative, 26 True Positives, it seems like the model gives a lot of false positives
confusion_matrix(y_test, y_pred) 

# Really starting score of bad score of .51% ... and then removing to the 5 features still gave me a low score of .525 ... Grid Searched helped increased to .6... and then I tried scaling but to no avail.
roc_auc_score(y_test, y_pred) 
