# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:53:08 2020

@author: Willie Man

Goals: Spam analysis


Planning: in text analytics in general, it seems to be that removing stop words, blanks then tokenizing, stemming, and then lemmatization.
    
Research/ Resources:
Great Project on text classification with different models: https://www.kaggle.com/anjanatiha/classification-with-multinomialnb-lstm-cnn-99
Text Classification Recipe: https://www.kaggle.com/naim99/text-classification-step-by-step
Reading Text files: https://www.kite.com/python/answers/how-to-read-a-text-file-with-pandas-in-python
read_csv documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
Removing stop words: https://stackabuse.com/removing-stop-words-from-strings-in-python/
Used this to update to dummy value: https://stackoverflow.com/questions/20250771/remap-values-in-pandas-column-with-a-dict
difference between TfidfVectorizer and CountVectorizer: https://datascience.stackexchange.com/questions/25581/what-is-the-difference-between-countvectorizer-token-counts-and-tfidftransformer#:~:text=The%20only%20difference%20is%20that,score%20while%20CountVectorizer()%20counts.
Questions: 
1. not familiar with tokenizer and countvectorizer


Future Endeavor:
    
    

"""



# Load Libraries:
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # converts text documents to numerical features.
from sklearn.feature_extraction.text import TfidfVectorizer # frequency of the word in all other documents
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import unicodedata
import sys

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split



# Input data:
data = pd.read_csv('SMSSpamCollection.txt', sep = '\t', header = None, names = ['target', 'text'])

# Preprocessing

## Removing Punctuations
## Create a dictionary of punctuation characters
punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
data['text'] = [string.translate(punctuation) for string in data['text']]

## Removing Stop Words (They don't hold value for classification)
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) # code borrowed from Python Cookbook

## Tokenize 
#data['text'] = data['text'].apply(lambda x: word_tokenize(x)) # Tokenizing breaks for some reason. it works better when i put under the count_vectorizer

## Applying Dummary Variables onto the label.
dummy_map = {'ham':0, 'spam':1}
data['target'] = data['target'].map(dummy_map)


# Feature Selection
X = data['text']
y = data['target']

#vect = CountVectorizer(tokenizer = word_tokenize) # added the tokenizer to the vectorizer. I think it expands the sparse matrix.
vect = TfidfVectorizer(min_df=3) # Can swap out to a different vectorizer.

X = vect.fit_transform(X)

# Split dataset:
X_train, X_test, Y_train, Y_test = train_test_split(X, y)

# Build the Model
clf = MultinomialNB()
clf = clf.fit(X_train, Y_train) 

# Evaluate the Model:
clf.score(X_train, Y_train)
clf.score(X_test, Y_test)

predictions = clf.predict(X_test)
accuracy_score(Y_test, predictions)
roc_auc_score(Y_test, predictions) # text does not work as a target. ROC needs dummy variables such as 0 and 1
classification_report(Y_test, predictions)

"""
Evaluation Scores: 
1st round: Train score is .993 and .977 which is higher than I would expect. seems to good to be true... this makes me skeptic. I did not token or stem in this one.

"""


""" Scoping Projects """


# Load libraries
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Create text
text_data = np.array(['I love Brazil. Brazil!',
'Brazil is best',
'Germany beats both'])

# Create bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Create feature matrix
features = bag_of_words.toarray()

# Create target vector
target = np.array([0,0,1]) # 0 is pro-brazil and 1 is pro-germany

# Create multinomial naive Bayes object with prior probabilities of each class
classifer = MultinomialNB(class_prior=[0.25, 0.5])

# Train model
model = classifer.fit(features, target)

# Create new observation
new_observation = [[0, 0, 0, 1, 0, 1, 0]]

# Predict new observation's class
model.predict(new_observation)


