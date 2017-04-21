import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from sklearn.ensemble import RandomForestClassifier

def meanAcc(estimator, X, y):
    predictions = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, predictions)
    cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    return cm_normalized.trace()/4


X_train = pd.read_csv('../data/allFeatures_standardized.csv')
X_test = pd.read_csv('../data/allYoutubeFeaturesStandardized.csv')

y_train = pd.read_csv('../data/allLabels.csv')
y_test = pd.read_csv('../data/emmotion_labels.csv')

assert sum(X_train['session'] != y_train['session']) == 0 # Ensure all sessions are the same
assert sum(X_test['session'] != y_test['session']) == 0 # Ensure all sessions are the same

X_train = X_train.drop('session',axis=1)
X_test = X_test.drop('session',axis=1)

y_train = np.ravel(y_train.drop(['time','session'],axis=1))
y_test = np.ravel(y_test.drop(['session'],axis=1))

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(0)

X_train = X_train.drop(['max(zerocrossing(zerocrossing))',
            'mean(zerocrossing(zerocrossing))'],axis=1)
X_test = X_test.drop(['max(zerocrossing(zerocrossing))',
            'mean(zerocrossing(zerocrossing))'],axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

svm = SVC(class_weight='balanced',C=50,gamma=0.03)
svm.fit(X_train_scaled,y_train)

train_predicions = svm.predict(X_train_scaled)
test_predictions = svm.predict(X_test_scaled)

cnf_matrix = confusion_matrix(y_train, train_predicions)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cm_normalized = cm_normalized.round(3) * 100
print("Svm train accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions

cnf_matrix = confusion_matrix(y_test, test_predictions)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cm_normalized = cm_normalized.round(3) * 100
print("Svm test accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions


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
