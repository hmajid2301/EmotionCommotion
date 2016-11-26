#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:12:34 2016

@author: olly
"""

# In[1]
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis




X = pd.read_csv('../data/allFeatures.csv')
#X = pd.read_csv('../features/mfcc.csv')

y = pd.read_csv('../data/allLabels.csv')

#X = X.merge(y).drop(['time','label'],axis=1)


assert sum(X['session'] != y['session']) == 0 # Ensure all sessions are the same

X = X.drop('session',axis=1)
y = np.ravel(y.drop(['session','time'],axis=1))

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(0)

def meanAcc(estimator, X, y):
    preds = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, preds)
    cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    return cm_normalized.trace()/4

min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
           

X_train = X_scaled[:3548]
X_test = X_scaled[3548:]
y_train = y[:3548]
y_test = y[3548:]

# In[2]

import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif_(X):

    '''X - pandas dataframe'''
    thresh = 5.0
    variables = list(range(X.shape[1]))

    for i in np.arange(0, len(variables)):
        vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            del variables[maxloc]

    print('Remaining variables:')
    print(X.columns[variables])
    return X[variables]

new_X = calculate_vif_(pd.DataFrame(X))

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(new_X, y).transform(new_X)

colors = ['red', 'turquoise', 'orange','green']
lw = 2

target_names = ['neu','ang','sad','hap']

plt.figure(figsize=(18,10))
for color, i, target_name in zip(colors, [0, 1, 2,3], target_names):
    plt.scatter(X_r2[y == target_name, 0], X_r2[y == target_name, 1], alpha=.5, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA')

plt.show()

# In[3]
colors = ['black', 'red', 'turquoise','yellow']


plt.figure(figsize=(18,12))
for color, i, target_name in zip(colors, [0, 1, 2,3], target_names):
    plt.scatter(X_r2[y == target_name, 0], X_r2[y == target_name, 1], alpha=.5, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1,fontsize=20)
plt.title('LDA',fontsize=20)

plt.show()
