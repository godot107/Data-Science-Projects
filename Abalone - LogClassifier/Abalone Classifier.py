# -*- coding: utf-8 -*-
"""
Logistic Regression Practice:


Created on Sun Nov  8 20:19:47 2020

@author: Willie Man

Research/References:

Datasource: https://archive.ics.uci.edu/ml/datasets/Abalone

Add column names: https://www.geeksforgeeks.org/add-column-names-to-dataframe-in-pandas/

SKLearn Confusion Matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
SKLearn ROC Curve: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
Label Encoding: https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/
label decoding: https://stackoverflow.com/questions/42196589/any-way-to-get-mappings-of-a-label-encoder-in-python-pandas
Interpretting ROC: https://www.displayr.com/what-is-a-roc-curve-how-to-interpret-it/
ROC Scoring Evaluation: https://www.researchgate.net/post/What_is_the_value_of_the_area_under_the_roc_curve_AUC_to_conclude_that_a_classifier_is_excellent#:~:text=for%20Atomic%20Research-,What%20is%20the%20value%20of%20the%20area%20under%20the%20roc,1%20denotes%20an%20excellent%20classifier.
Logistic Regression walkthrough: https://towardsdatascience.com/logistic-regression-a-simplified-approach-using-python-c4bc81a87c31

"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Current directory:
os.chdir('C:\\Users\Willie Man\OneDrive\Personal Projects\Python\Projects\Abalone - Classifier')

# Import data:
data = pd.read_csv('abalone.csv', header = None)
# Call this method to rename the columns according to the documentation
data.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',  'Viscera weight ','Shell weight ', 'Rings']
# Removing the infant for 'Sex' that way we get binary results.
data = data[data['Sex'] != 'I']

# 


# Feature Extraction
Features = ['Length', 'Diameter',  'Height','Whole weight','Shucked weight', 'Viscera weight ','Shell weight ','Rings']
#Features = ['Length', 'Diameter',  'Height']

# Feature Selection:
X = data[Features]
y = data['Sex']


# Scaling the Features, since they are different measurements and scale.
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)  # LabelEncoder converts the string to numeric so it can be calculated in sklearn. 1 = Male and 0 = Female

# This helps you map back from encoder to the label.
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))



# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, random_state=0) 

# Building the Model:
log_reg = LogisticRegression()

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]} # incorporating grid search and cycle through the hyperparameters
logreg_cv=GridSearchCV(log_reg,grid,cv=10)


# Train Model:
model = logreg_cv.fit(X_train, y_train)

# Testing/Evaluating the Model:
model.score(X_standardized,y)

predictions = model.predict(X_test)
predictions_prob = model.predict_proba(X_test)
confusion_matrix(y_test, predictions)
roc_auc_score(np.array(y_test), predictions) 

# more evaluation to our model's performance.
print(classification_report(y_test,predictions))

"""
Remarks:
ROS score is low at around .533  and maybe because all the features are numeric and not categorical. maybe this is not the best model to choose from.

I tried to do feature selection to try and narrow it down, but not successful. perhaps, log is not the best model?

"""





