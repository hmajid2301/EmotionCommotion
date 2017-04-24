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

clf_funcs = [get_gnb, get_svm, get_rf]
clf_names = ['gnb', 'svm', 'rf']

for i in range(0,len(clf_funcs)):
	clf = clf_funcs[i]()
	clf_name = clf_names[i]
	clf.fit(X_train_scaled,y_train)
	train_predicions = clf.predict(X_train_scaled)
	test_predictions = clf.predict(X_test_scaled)

	# accuracy = save_confusion_matrix(y_train, train_predicions, 'confusion_matrices/' + clf_name + "_IEMOCAP.png")
	# print(clf_name + " train accuracy: ", accuracy)  # Average accuracy accross all 4 emotions


	accuracy = save_confusion_matrix(y_test, test_predictions, 'confusion_matrices/' + clf_name + "_YouTube.png")
	print(clf_name + " test accuracy: ", accuracy)  # Average accuracy accross all 4 emotions
