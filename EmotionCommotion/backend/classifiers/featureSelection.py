from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

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



lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape
