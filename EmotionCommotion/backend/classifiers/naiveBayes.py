# -*- coding: utf-8 -*-

# In[1]
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.naive_bayes import GaussianNB


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

import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, title='Confusion matrix - Naive Bayes', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(['Angry','Happy','Neutral','Sad']))
    plt.xticks(tick_marks, ['Angry','Happy','Neutral','Sad'], rotation=45)
    plt.yticks(tick_marks, ['Angry','Happy','Neutral','Sad'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.savefig("../results/naiveBayesCM.png")


plot_confusion_matrix(cm_normalized)