# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 21:41:23 2020

@author: Willie Man

Goal: Predict the MEDV 
Data Source: https://www.kaggle.com/arslanali4343/real-estate-dataset


Research/Resources:
No Scaling needed for Decision Tree: https://stackoverflow.com/questions/29842647/feature-scaling-required-or-not    
DecisionTreeRegressor Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
General understanding of decision Trees: https://www.analyticsvidhya.com/blog/2020/06/4-ways-split-decision-tree/
Pre-processing: https://www.geeksforgeeks.org/how-to-drop-rows-with-nan-values-in-pandas-dataframe/
Feature Importance: https://machinelearningmastery.com/calculate-feature-importance-with-python/



Future Endeavor:

Observation: 
1. this data set is pretty racist because it actually keeps track of the black population. This data set is titled 'Concerns housing values in suburbs of Boston'  HOLY SHIT. but it is an older study from 1993. but still.   
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from IPython.display import Image
from sklearn.tree import export_graphviz
# Import the data:
data = pd.read_csv('data.csv')

# Pre-processing
data = data.dropna() # drop records if there is a NaN in the row.

# Feature Selection
# X = data.iloc[:,np.arange(13)].values // if i decide to run the model with all features.
X = data.iloc[:,[5,12]].values  # selected based on feature selection
y_target = data['MEDV']


# get importance

model = DecisionTreeRegressor(max_depth=2).fit(X, y_target)
importance = model.feature_importances_

for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

# based on this feature 5 and 12 are the importance ones.

# Feature Scaling - not necessary for Decision Trees

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_target)

# Build the Model
regressor = DecisionTreeRegressor(max_depth=2)
regressor  = regressor.fit(X_train, y_train)

# Build Predictions:
predictions = regressor.predict(X_test)

# Evaluate the Model
# Evaluating + Exploring
r2_score(y_test,predictions) # this is r^2 coefficient.

"""
1st attempt of r2_score with ALL features is at .63
2nd attempt of r2_score with less features, but important is at .56
"""

regressor.score(X, y_target)




# Viz




""" REFERENCE """
