import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from classifiers import *

test_index = 3548


X = pd.read_csv('../data/allFeatures_standardized.csv')
y = pd.read_csv('../data/allLabels.csv')

assert sum(X['session'] != y['session']) == 0 # Ensure all sessions are the same

X = X.drop('session',axis=1)
y = np.ravel(y.drop(['time','session'],axis=1))

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = X.drop(['max(zerocrossing(zerocrossing))',
            'mean(zerocrossing(zerocrossing))'],axis=1)

X_train = X[:test_index]
X_test = X[test_index:]
y_train = y[:test_index]
y_test = y[test_index:]

min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

clf_funcs = [get_gnb, get_svm, get_rf]
clf_names = ['gnb', 'svm', 'rf']

for i in range(0,len(clf_funcs)):
	clf = clf_funcs[i]()
	clf_name = clf_names[i]
	clf.fit(X_train_scaled,y_train)
	train_predicions = clf.predict(X_train_scaled)
	test_predictions = clf.predict(X_test_scaled)

	accuracy = save_confusion_matrix(y_test, test_predictions, 'confusion_matrices/' + clf_name + "_iemocap.png")
	print(clf_name + " test accuracy: ", accuracy)  # Average accuracy accross all 4 emotions
