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

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import GroupKFold
import itertools
from sklearn.externals import joblib

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

# In[2]

#lda = LinearDiscriminantAnalysis(n_components=12)
#new_X = calculate_vif_(pd.DataFrame(X))
#X = lda.fit(new_X, y).transform(new_X)

# Scale features
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
           

X_train = X_scaled[:3548]
X_test = X_scaled[3548:]
y_train = y[:3548]
y_test = y[3548:]

groups =(   [1 for _ in range(0,942)] +
            [2 for _ in range(942,1755)] + 
            [3 for _ in range(1755,2755)] + 
            [4 for _ in range(2755,3548)] 
        )

gkf = GroupKFold(n_splits=4)
fold_iter = gkf.split(X_train, y_train, groups=groups)

tuned_parameters = [{'C': [50,100,200,500,1000]}]
                    
svm = GridSearchCV(SVC(class_weight='balanced'), tuned_parameters, cv=fold_iter,verbose=10,scoring=meanAcc, n_jobs = -1)
svm.fit(X_train, y_train)

preds = svm.predict(X_test)

cnf_matrix = confusion_matrix(y_test, preds)
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
    plt.savefig("../results/SVMCM.png",transparent=True,figsize=(20,20),dpi=120)


plot_confusion_matrix(cm_normalized,fontsize=14)

    