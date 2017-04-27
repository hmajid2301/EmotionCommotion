# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:56:53 2016

@author: olly
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import GroupKFold
import itertools
from sklearn.externals import joblib

SESSION_1_END = 942
SESSION_2_END = 1755
SESSION_3_END = 2755
SESSION_4_END = 3548
TRAIN_END = 3548


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
    predictions = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, predictions)
    cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    return cm_normalized.trace()/4


X = X.drop(['max(zerocrossing(zerocrossing))',
            'mean(zerocrossing(zerocrossing))'],axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)


X_train = X_scaled[:TRAIN_END]
X_test = X_scaled[TRAIN_END:]
y_train = y[:TRAIN_END]
y_test = y[TRAIN_END:]

groups =(   [1 for _ in range(0,SESSION_1_END)] +
            [2 for _ in range(SESSION_1_END,SESSION_2_END)] +
            [3 for _ in range(SESSION_2_END,SESSION_3_END)] +
            [4 for _ in range(SESSION_3_END,SESSION_4_END)]
        )

gkf = GroupKFold(n_splits=4)
fold_iter = gkf.split(X_train, y_train, groups=groups)

param_grid = [{'C': [3,5,10,20,30,40,50],'gamma':[0.01,0.03,0.05,0.1,0.2,0.3]}]

svm = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=fold_iter,verbose=10,scoring=meanAcc, n_jobs = -1)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)

cnf_matrix = confusion_matrix(y_test, predictions)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cm_normalized = cm_normalized.round(3) * 100


print('Best hyperparameters: %r' % svm.best_params_)

print("Accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions
joblib.dump(svm.best_estimator_, 'svm.pkl')



# In[5]
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, title='SVM Confusion matrix', cmap=plt.cm.Greens,fontsize=14):
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
    #plt.savefig("../results/SVMCM.png",transparent=True,figsize=(20,20),dpi=120)


plot_confusion_matrix(cm_normalized,fontsize=14)
