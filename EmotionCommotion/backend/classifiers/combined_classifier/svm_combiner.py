from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
X = pd.read_csv('./IEMOCAP_frame_agg.csv')
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
y = pd.read_csv('../../data/allLabels.csv')
print(X.shape)
print(y.shape)
#Drop unnecessary columns
y = np.ravel(y.drop(['session','time'],axis=1))
# 3548 is the first sample from session 5
X_train = X
X_test = pd.read_csv('./wild_frame_agg.csv')
y_train = y
y_test = pd.read_csv('./wild_dataset/wild_dataset/emotion_labels.csv')
y_test = np.ravel(y_test.drop(['filename'],axis=1))
print(X_test.shape)
print(y_test.shape)
# X_train = X[:3548]
# X_test = X[3548:]
# y_train = y[:3548]
# y_test = y[3548:]

# X_train = np.array(X_train)
# X_test = np.array(X_test)
# X_train = np.delete(X_train,[0,1,8,9,10],1)
# X_test = np.delete(X_test,[0,1,8,9,10],1)
# Scoring metric
def meanAcc(estimator, X, y):
    preds = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, preds)
    cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    return cm_normalized.trace()/4

# groups =(   [1 for _ in range(0,942)] +
#             [2 for _ in range(942,1755)] + 
#             [3 for _ in range(1755,2755)] + 
#             [4 for _ in range(2755,3548)] 
#         )

# gkf = GroupKFold(n_splits=4)
# fold_iter = gkf.split(X_train, y_train, groups=groups)

param_grid = [
  #{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
 ]

# param_grid = [
#   {'C': [100], 'kernel': ['linear']}
#  ]

# groups =(   [1 for _ in range(0,942)] +
#             [2 for _ in range(942,1755)] + 
#             [3 for _ in range(1755,2755)] + 
#             [4 for _ in range(2755,3547)] 
#         )

# gkf = GroupKFold(n_splits=4)
# fold_iter = gkf.split(X_train, y_train, groups=groups)

tuned_parameters = [{'C': [50,100,200,500,1000]}]
tuned_parameters = [{'C': [53,103,203]}]
tuned_parameters = [{'C': [203]}]

# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]

svm = GridSearchCV(SVC(class_weight='balanced'), tuned_parameters, cv=5,verbose=5,scoring=meanAcc, n_jobs = -1)
svm.fit(X_train, y_train)

preds = svm.predict(X_test)
# for i in range(10,20):
#     print(X_test[i-1:i])
#     print(preds[i-1:i])
#     print(y_test[i-1:i])
cnf_matrix = confusion_matrix(y_test, preds)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cm_normalized = cm_normalized.round(3) * 100
print(svm.best_params_)
print("Accuracy: ", cm_normalized.trace()/4)

import matplotlib.pyplot as plt
import itertools
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
    plt.savefig("./svm_time_wsum.png",transparent=True,figsize=(20,20),dpi=120)


plot_confusion_matrix(cm_normalized)
