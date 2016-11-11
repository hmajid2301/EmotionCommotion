#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:56:53 2016

@author: olly
"""

# In[1]
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np




X = pd.read_csv('../data/allFeatures.csv')
y = pd.read_csv('../data/allLabels.csv')

assert sum(X['session'] != y['session']) == 0 # Ensure all sessions are the same

X = X.drop('session',axis=1)
y = np.ravel(y.drop(['session','time'],axis=1))

min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train)
preds = logreg.predict(X_test)

cnf_matrix = confusion_matrix(y_test, preds)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print("Accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions

