# -*- coding: utf-8 -*-

# In[1]
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import itertools

X = pd.read_csv('../data/allFeatures.csv')

y = pd.read_csv('../data/allLabels.csv')


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

# Scale features
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)


X_train = X_scaled[:3548]
X_test = X_scaled[3548:]
y_train = y[:3548]
y_test = y[3548:]

gnb = GaussianNB()

gnb.fit(X_train, y_train)

preds = gnb.fit(X_train, y_train).predict(X_test)

cnf_matrix = confusion_matrix(y_test, preds)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print("Accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions
cm_normalized = cm_normalized.round(3) * 100

def plot_confusion_matrix(cm, title='NB Confusion matrix', cmap=plt.cm.Greens,fontsize=14):
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
    plt.savefig("../results/NBCM.png",transparent=True,figsize=(20,20),dpi=120)



plot_confusion_matrix(cm_normalized)
plt.show()