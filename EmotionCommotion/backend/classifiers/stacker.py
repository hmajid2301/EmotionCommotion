import pandas as pd
import numpy as np
from classifiers import *
import pickle 

# Get classifier tuned previously using grid search
def get_stacker_probs(model_func, X_train, y_train, X_test, split_index):
	# Split training data by index
	X_train_a = X_train[:split_index]
	X_train_b = X_train[split_index:]
	y_train_a = y_train[:split_index]
	y_train_b = y_train[split_index:]
	# Fit model a using first half of data
	model_a = model_func()
	model_a.fit(X_train_a, y_train_a)
	# Use model a to predict on the second half of data
	train_probs_b = model_a.predict_proba(X_train_b)
	# Fit model b using second half of data
	model_b = model_func()
	model_b.fit(X_train_b, y_train_b)
	# Use model b to predict on the first half of data
	train_probs_a = model_b.predict_proba(X_train_a)
	# Combine predictions
	train_probs = np.concatenate((train_probs_a,train_probs_b),axis=0)
	# Fit final model using all of data
	model_c = model_func()
	model_c.fit(X_train,y_train)
	test_probs = model_c.predict_proba(X_test)
	return [model_c, train_probs, test_probs]



# read in features extracted from datasets
X_train = pd.read_csv('../data/allFeatures_standardized.csv')
X_test = pd.read_csv('../data/allYoutubeFeaturesStandardized.csv')
# Read in labels
y_train = pd.read_csv('../data/allLabels.csv')
y_test = pd.read_csv('../data/emmotion_labels.csv')
# Ensure session names match
assert sum(X_train['session'] != y_train['session']) == 0 # Ensure all sessions are the same
assert sum(X_test['session'] != y_test['session']) == 0 # Ensure all sessions are the same
# Drop unecessary values
X_train = X_train.drop('session',axis=1)
X_test = X_test.drop('session',axis=1)
y_train = np.ravel(y_train.drop(['time','session'],axis=1))
y_test = np.ravel(y_test.drop(['session'],axis=1))
# Fill in infinite and nan values with 0s
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(0)

# Dropping  features deemed unuseful from previous tests
X_train = X_train.drop(['max(zerocrossing(zerocrossing))',
            'mean(zerocrossing(zerocrossing))'],axis=1)
X_test = X_test.drop(['max(zerocrossing(zerocrossing))',
            'mean(zerocrossing(zerocrossing))'],axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

# Classifiers to stack
clf_funcs = [get_gnb, get_svm, get_rf]
clf_names = ['gnb', 'svm', 'rf']

# Index of final sample from session 3
split_index = 3548

for i in range(0,len(clf_funcs)):
	clf_func = clf_funcs[i]
	clf_name = clf_names[i]
	print("Stacking " + clf_name + "...")
	[model, train_probs, test_probs] = get_stacker_probs(clf_func, X_train, y_train, X_test, split_index)
	np.savetxt('probs/' + clf_name + '_train.csv', train_probs, delimiter=',')
	np.savetxt('probs/' + clf_name + '_test.csv', test_probs, delimiter=',')
	pickle.dump(model, open('saved_classifiers/' + clf_name + '.pkl', 'wb'))