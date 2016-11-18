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
from sklearn.ensemble import GradientBoostingClassifier


from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np




X = pd.read_csv('../data/allFeatures.csv')
y = pd.read_csv('../data/allLabels.csv')

assert sum(X['session'] != y['session']) == 0 # Ensure all sessions are the same

y = np.ravel(y.drop(['session','time'],axis=1))

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(0)

X = X.drop('session',axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)



# 3548 is the first sample from session 5
X_train = X[:3548]
X_test = X[3548:]
y_train = y[:3548]
y_test = y[3548:]

param_grid = {'learning_rate': [0.1, 0.01, 0.001],
              'max_depth': [4, 6],
              'min_samples_leaf': [3, 5],  ## depends on the nr of training examples
              'max_features': [1.0, 0.3, 0.1] ## not possible in our example (only 1 fx)
              }
    

est = GradientBoostingClassifier(n_estimators=2000)
# this may take some minutes
gs_cv = GridSearchCV(est, param_grid, n_jobs=-1,verbose=10).fit(X_train, y_train)

# best hyperparameter setting
print('Best hyperparameters: %r' % gs_cv.best_params_)

                    
gs_cv.fit(X_train, y_train)

preds = gs_cv.predict(X_test)

cnf_matrix = confusion_matrix(y_test, preds)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print("Accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions


est.set_params(**gs_cv.best_params_)
est.fit(X_train, y_train)
# In[3]
fx_imp = pd.Series(est.feature_importances_, index=X.columns)
fx_imp /= fx_imp.max()  # normalize
fx_imp.sort()
fx_imp.plot(kind='barh',figsize=(10,10))

# In[4]
from sklearn.ensemble.partial_dependence import plot_partial_dependence

fig, axs = plot_partial_dependence(est, X_train, list(X.columns), feature_names=list(X.columns), 
                                   n_cols=2,label='neu',figsize=(10,20))
# In[5]
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(['Angry','Happy','Neutral','Sad']))
    plt.xticks(tick_marks, ['Angry','Happy','Neutral','Sad'], rotation=45)
    plt.yticks(tick_marks, ['Angry','Happy','Neutral','Sad'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(cm_normalized)