import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

def meanAcc(estimator, X, y):
    predictions = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, predictions)
    cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    return cm_normalized.trace()/4

def plot_confusion_matrix(cm, title='CNN Confusion matrix', cmap=plt.cm.Greens,
                          fontsize=14,filename="confusion_matrix.png"):
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
    plt.savefig(filename,transparent=True,figsize=(20,20),dpi=120)


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

# Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train_scaled,y_train)

train_predicions = gnb.predict(X_train_scaled)
test_predictions = gnb.predict(X_test_scaled)

cnf_matrix = confusion_matrix(y_train, train_predicions)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cm_normalized = cm_normalized.round(3) * 100
print("NB train accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions

cnf_matrix = confusion_matrix(y_test, test_predictions)
cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cm_normalized = cm_normalized.round(3) * 100
print("NB test accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions

plot_confusion_matrix(cm_normalized,fontsize=14,title='SVM Confusion Matrix',
                      filename='../results/yt_gnb_cm.png')
# SVM

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

plot_confusion_matrix(cm_normalized,fontsize=14,title='SVM Confusion Matrix',
                      filename='../results/yt_svm_cm.png')

# Random Forest

rf = RandomForestClassifier(n_estimators=2000,max_depth=18,max_features=0.3,min_samples_leaf=0.005)
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

plot_confusion_matrix(cm_normalized,fontsize=14,title='RF Confusion Matrix',
                      filename='../results/yt_rf_cm.png')
