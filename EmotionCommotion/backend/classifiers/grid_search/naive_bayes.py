# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import itertools

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

# Scale features
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

# Split into training and testing
X_train = X_scaled[:TRAIN_END]
X_test = X_scaled[TRAIN_END:]
y_train = y[:TRAIN_END]
y_test = y[TRAIN_END:]

# Use Naive Bayes to make predictions
gnb = GaussianNB()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predictions)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print("Accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions
cm_normalized = cm_normalized.round(3) * 100

def plot_confusion_matrix(cm, title='NB Confusion matrix', cmap=plt.cm.Greens,fontsize=14):
    '''
    Given a confusion matrix, plots the values and saves the fuigure
    '''
    # Set figure size
    plt.figure(figsize=(4.5,4.5))

    # Set other aethetics
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=fontsize+3)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize)
    tick_marks = np.arange(len(['Angry','Happy','Neutral','Sad']))
    plt.xticks(tick_marks, ['Angry','Happy','Neutral','Sad'], rotation=45,fontsize=fontsize)
    plt.yticks(tick_marks, ['Angry','Happy','Neutral','Sad'],fontsize=fontsize)
    thresh = cm.max() / 1.5

    # Overlay numbers on confusion matrix
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(cm[i, j]) + "%",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                    fontsize=fontsize)

    # Set labels and save figure
    plt.tight_layout()
    plt.ylabel('True label',fontsize=fontsize)
    plt.xlabel('Predicted label',fontsize=fontsize)
    plt.gcf().subplots_adjust(bottom=0.25,left=0.25)
    plt.savefig("../results/NBCM.png",transparent=True,figsize=(20,20),dpi=120)



plot_confusion_matrix(cm_normalized)
plt.show()
