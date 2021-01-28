# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:47:17 2020

@author: Willie Man

Research/Resources:
    
Scikit learn documentation - https://scikit-learn.org/stable/modules/svm.html

Scoping Projects - https://www.kaggle.com/bhuvaneshwaran/rental-listing-inquiries-linearsvc

Tutorial (useful): https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

Iris dataset: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

Why i need to standardize x-value: https://stats.stackexchange.com/questions/65094/why-scaling-is-important-for-the-linear-svm-classification

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




# Get Features (X) and Target (y)
iris = datasets.load_iris()
features = iris.feature_names

X = pd.DataFrame(iris.data, columns = features)
y = iris.target
target_names = iris.target_names

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)


# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, random_state=0) 


# Build our SVM classifier:

clf = SVC(kernel='linear')

# Grid Search (optional)
param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear']}
clf = GridSearchCV(clf, param_grid)

# clf = OneVsRestClassifier(clf) an attempt for ROC score.

# Best combination of parameters to use based on the gridsearch
#clf.best_params_

# Train the model
clf.fit(X_train, y_train)


# Predictions:
y_pred = clf.predict(X_test)


# Evaluate the model:
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Confusion Matrix
#confusion_matrix(y_test, y_pred)

print(classification_report(y_test,y_pred))

#roc_auc_score(y_test, y_pred, multi_class="ovo") // Tried to do AUC score, but it is struggle due to multi-class classifier, meaning the output isn't binary.





