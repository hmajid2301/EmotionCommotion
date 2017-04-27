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

def save_confusion_matrix(y_test, test_predictions, filename='tmp.png'):
	cnf_matrix = confusion_matrix(y_test, test_predictions)
	cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
	cm_normalized = cm_normalized.round(3) * 100

	plot_confusion_matrix(cm_normalized,fontsize=14,title='Ensemble Confusion Matrix',
                      filename=filename)
	return cm_normalized.trace()/4 # Average accuracy accross all 4 emotions

# Get classifier tuned previously using grid search
def get_gnb():
	gnb = GaussianNB()
	return gnb

# Get classifier tuned previously using grid search
def get_svm():
	svm = SVC(class_weight='balanced',C=50,gamma=0.03, probability = True)
	return svm

# Get classifier tuned previously using grid search
def get_rf():
	rf = RandomForestClassifier(n_estimators=2000,max_depth=18,max_features=0.3,min_samples_leaf=0.005)
	return rf
