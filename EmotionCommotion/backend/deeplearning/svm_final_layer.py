import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


cnn_outputs_train = np.load('cnn_outputs_train.npy')
cnn_outputs_test = np.load('cnn_outputs_test.npy')

features_train = np.load('../features/all_framewise_train.npy')
features_test = np.load('../features/all_framewise_test.npy')

X_train = np.concatenate((cnn_outputs_train,features_train),axis=1)
X_test = np.concatenate((cnn_outputs_test,features_test),axis=1)
y_train = np.load('../../../../local/whitened_data/y_train.npy')
y_test = np.load('../../../../local/whitened_data/y_test.npy')

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

y_train = np.argmax(y_train,axis=1)
y_test = np.argmax(y_test,axis=1)

clf = SVC()
clf.fit(X_train,y_train) 
preds = clf.predict(X_test)
cnf_matrix = confusion_matrix(y_test, preds)

cm_normalized = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cm_normalized = cm_normalized.round(3) * 100


print("Accuracy: ", cm_normalized.trace()/4)  # Average accuracy accross all 4 emotions

