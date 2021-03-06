# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:56:53 2016

@author: olly
"""

# In[1]
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn_evaluation import plot

# 3548 is the first sample from session 5
TRAIN_END = 3548


# Load features
X = pd.read_csv('../data/allFeatures.csv')

# Load labels
y = pd.read_csv('../data/allLabels.csv')

# Ensure all sessions are the same
assert sum(X['session'] != y['session']) == 0

# Drop session name from features
X = X.drop('session',axis=1)

# Remove session name and time from labels
y = np.ravel(y.drop(['session','time'],axis=1))

# Replace nans and infitities with 0s
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# Scoring metric
def meanAcc(estimator, X, y):
    '''
    Returns the mean accuracy of predictions, weighted inversely proportionally to the number
    of samples in each class.
    '''
    # Use estimator to make predictions
    predictions = estimator.predict(X)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y, predictions)
    # Normalize confusion matrix
    cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    # Return mean accuracy
    return cm_normalized.trace()/4

# Groups for cross validation. Each group is a different speaker
groups =(   [1 for _ in range(0,942)] +
            [2 for _ in range(942,1755)] +
            [3 for _ in range(1755,2755)] +
            [4 for _ in range(2755,3548)]
        )


# Transform all features to be in the range 0-1
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)


# Split into training and testing
X_train = X[:TRAIN_END]
X_test = X[TRAIN_END:]
y_train = y[:TRAIN_END]
y_test = y[TRAIN_END:]


X_train = X_train.drop(['max(zerocrossing(zerocrossing))',
            'mean(zerocrossing(zerocrossing))'],axis=1)
X_test = X_test.drop(['max(zerocrossing(zerocrossing))',
            'mean(zerocrossing(zerocrossing))'],axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)


rf = RandomForestClassifier(n_estimators=1000,max_depth=18,max_features=0.3,min_samples_leaf=0.005)
rf.fit(X_train_scaled,y_train)

train_predicions = rf.predict(X_train_scaled)
test_predictions = rf.predict(X_test_scaled)

cnf_matrix = confusion_matrix(y_train, train_predicions)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cm_normalized = cm_normalized.round(3) * 100
print("rf train accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions

cnf_matrix = confusion_matrix(y_test, test_predictions)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cm_normalized = cm_normalized.round(3) * 100
print("rf test accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions

# X_test = lda.transform(X_test)

# Parameters to try for grid search
param_grid = {'max_depth': [8,16,32,64],
              'min_samples_leaf': [0.001, 0.005,0.01,0.02],
              'max_features': [1.0, 0.3, 0.1]
              }
param_grid = {'max_depth': [None]          }


est = RandomForestClassifier()
# Run gridsearch
gs_cv = GridSearchCV(est, param_grid, n_jobs=-1,verbose=10,scoring=meanAcc).fit(X_train, y_train)

# best hyperparameter setting
print('Best hyperparameters: %r' % gs_cv.best_params_)


# Make predictions on test set
predictions = gs_cv.predict(X_test)

cnf_matrix = confusion_matrix(y_test, predictions)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print("Accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions


cm_normalized = cm_normalized.round(3) * 100

def plot_confusion_matrix(cm, title='RF Confusion matrix', cmap=plt.cm.Greens,fontsize=14):
    plt.figure(figsize=(4.5,4.5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=fontsize+3)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize)

    tick_marks = np.arange(len(['Angry','Happy','Neutral','Sad']))
    plt.xticks(tick_marks, ['Angry','Happy','Neutral','Sad'], rotation=45,fontsize=fontsize)
    plt.yticks(tick_marks, ['Angry','Happy','Neutral','Sad'],fontsize=fontsize)
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(cm[i, j]) + "%",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                    fontsize=fontsize)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=fontsize)
    plt.xlabel('Predicted label',fontsize=fontsize)
    plt.gcf().subplots_adjust(bottom=0.25,left=0.25)
    plt.savefig("../results/rf_cm_before_grid.png",transparent=True,figsize=(20,20),dpi=120)


plot_confusion_matrix(cm_normalized)
plt.show()
# plot.grid_search(gs_cv.grid_scores_, change=('max_depth', 'min_samples_leaf'), subset={'max_features': 1.0})
# plt.show()
# plot.grid_search(gs_cv.grid_scores_, change=('max_depth', 'min_samples_leaf'), subset={'max_features': 0.3})
# plt.show()
# plot.grid_search(gs_cv.grid_scores_, change=('max_depth', 'min_samples_leaf'), subset={'max_features': 0.1})
# plt.show()
