from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import argparse
import classifiers
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from classifiers import *
import pickle

# Read in the aggregated features provided by the cnn on the IEMOCAP database
features_train = pd.read_csv('../data/allFeatures_standardized.csv')
features_test = pd.read_csv('../data/allYoutubeFeaturesStandardized.csv')
features_train = features_train.replace([np.inf, -np.inf], np.nan)
features_test = features_test.replace([np.inf, -np.inf], np.nan)


# pca = PCA(n_components=8)
# pca.fit(features_train)
# features_train = pca.transform(features_train)
# features_test = pca.transform(features_test)

#features = features.fillna(0)
#features = features.iloc[:,[0,1,4,10,16,20,31,33,37,42,43,44,45,46,50,55,57,58,61,63,65]]

# X = features.iloc[:,:8]



#nb = pd.read_csv('./probs/classic_nb.csv')
svm = pd.read_csv('./probs/svm_train.csv', header = None)
rf = pd.read_csv('./probs/rf_train.csv', header = None)
cnn_svm = pd.read_csv('./probs/svmcnn_train.csv', header = None)
cnn_rf = pd.read_csv('./probs/rfcnn_train.csv', header = None)
#cnn_gnb = pd.read_csv('./probs/gnbcnn_train.csv', header = None)

svm = np.array(svm)
rf = np.array(rf)
cnn_svm = np.array(cnn_svm)
cnn_rf = np.array(cnn_rf)
print(svm.shape)
print(rf.shape)

X_train = np.concatenate((svm, rf), axis= 1)
X_train = np.concatenate((X_train, cnn_svm), axis= 1)
X_train = np.concatenate((X_train, cnn_svm), axis= 1)
#X_train = np.concatenate((X_train, cnn_gnb), axis= 1)
#X_train = np.concatenate((features_train, cnn_svm), axis= 1)

svm = pd.read_csv('./probs/svm_test.csv', header = None)
rf = pd.read_csv('./probs/rf_test.csv', header = None)
cnn_svm = pd.read_csv('./probs/svmcnn_test.csv', header = None)
cnn_rf = pd.read_csv('./probs/rfcnn_test.csv', header = None)
#cnn_gnb = pd.read_csv('./probs/gnbcnn_test.csv', header = None)


svm = np.array(svm)
rf = np.array(rf)
cnn_svm = np.array(cnn_svm)
cnn_rf = np.array(cnn_rf)

X_test = np.concatenate((svm, rf), axis= 1)
X_test = np.concatenate((X_test, cnn_svm), axis= 1)
X_test = np.concatenate((X_test, cnn_svm), axis= 1)
#X_test = np.concatenate((X_test, cnn_gnb), axis= 1)
#X_test = np.concatenate((features_test, cnn_svm), axis= 1)

y_train = pd.read_csv('../data/allLabels.csv')
y_test = pd.read_csv('../data/emmotion_labels.csv')
y_train = np.ravel(y_train.drop(['time','session'],axis=1))
y_test = np.ravel(y_test.drop(['session'],axis=1))

print(X_train.shape)
print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

# Test on IEOMOCAP
# X_test = X_train[3548:]
# X_train = X_train[:3548]
# y_test = y_train[3548:]
# y_train = y_train[:3548]


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)




# estimator = SVC(kernel='linear')#,class_weight='balanced', C=53)
# selector = RFE(estimator, 16, step=1, verbose=10)
# selector = selector.fit(X_train, y_train)
# print(selector.ranking_)
# X_train = selector.transform(X_train)
# X_test = selector.transform(X_test)

ensemble_model = get_svm()
ensemble_model.fit(X_train_scaled,y_train)
preds = ensemble_model.predict(X_test_scaled)
print(preds.shape)
accuracy = save_confusion_matrix(y_test, preds, filename='confusion_matrices/ensemble_wild.png')

print("YouTube Accuracy: ", accuracy)

iemo_X_train = X_train[:3548]
iemo_X_test = X_train[3548:]
iemo_y_train = y_train[:3548]
iemo_y_test = y_train[3548:]

min_max_scaler = preprocessing.MinMaxScaler()
iemo_X_train_scaled = min_max_scaler.fit_transform(iemo_X_train)
iemo_X_test_scaled = min_max_scaler.transform(iemo_X_test)

ensemble_model = get_svm()
ensemble_model.fit(iemo_X_train_scaled,iemo_y_train)
preds = ensemble_model.predict(iemo_X_test)
print(preds.shape)
accuracy = save_confusion_matrix(iemo_y_test, preds, filename='confusion_matrices/ensemble_iemocap.png')

print("IEMOCAP Accuracy: ", accuracy)

#pickle.dump(ensemble_model, open('saved_classifiers/ensemble.pkl', 'wb'))
