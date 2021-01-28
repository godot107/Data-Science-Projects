# -*- coding: utf-8 -*-
"""
Multiple Linear Regression Practice:
Attempting to predict final grade from first and second period grade.

Created on Sun Nov  8 17:24:38 2020

@author: Willie Man

Research/References:
    
Source of data: https://archive.ics.uci.edu/ml/datasets/Student+Performance
change directory: https://note.nkmk.me/en/python-os-getcwd-chdir/
matplots: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.scatter.html
matplots: https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html
Interpretting R: https://statisticsbyjim.com/regression/how-high-r-squared/#:~:text=Any%20study%20that%20attempts%20to,%2Dsquared%20values%20over%2090%25.
Interpretting RSME: https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e
Visualizing Multiple Linear Regression and Viz: https://aegis4048.github.io/mutiple_linear_regression_and_visualization_in_python
"""





import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


# Current directory:
#os.chdir('C:\\Users\Willie Man\OneDrive\Personal Projects\Python\Projects\Student Performance Linear Reg')


# Import data:

data = pd.read_csv('student-mat.csv', sep = ';')


# Feature Selection:

X = data[['G1', 'G2']]  # the x and y are in the same scale so it doesn't need to be standardized.
y = data[['G3']]

"""
31 G1 - first period grade (numeric: from 0 to 20)
31 G2 - second period grade (numeric: from 0 to 20)
32 G3 - final grade (numeric: from 0 to 20, output target)

"""

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) # random_state so we can get the same results with others.

# building our regression model
linreg = LinearRegression().fit(X_train, y_train)    

# Predict using X_test/X_train
test_predictions = linreg.predict(X_test)
train_predictions = linreg.predict(X_train)

# passing predictions based on model
x_pred = np.array([15,18])
x_pred = x_pred.reshape(-1,2) # reshape operator allows the x features to be passed
linreg.predict(x_pred) # passing the numpy array

# Evaluating + Exploring
r2_score(y_test,test_predictions) # this is r^2 coefficient.
r2 = linreg.score(X, y)

# Evaluating rmse and it looks like test_rmse > train_rmse by less than 1, which suggest that my model is overfitting. 
test_rmse = (np.sqrt(mean_squared_error(y_test, test_predictions)))
train_rmse = (np.sqrt(mean_squared_error(y_train, train_predictions)))


linreg.intercept_
linreg.coef_ # as G1 goes up, model predicts the final grade (G3) to go up as .095 and as G2 goes up, the model predicts G3 goes up by 1.0057.  So this shows that G2 has a slightly more influence on the final grade.

# Creating a visual:
plt.plot(y_test, test_predictions, 'o', color='black'); # generally same ball park as a 1 to 1

plt.plot(data[['G1']], data[['G2']], 'o', color='black'); # Observing and testing to see if G1 has a correlation with G2

plt.plot(X_test, y_test, 'o', color='red', alpha = .25); # The plot has no value because i have two features. The best way is to split them out.

plt.show()


"""
Remarks:
    
Looking at the graph between the two X values/features. we can see a positive correlation.
    
"""




