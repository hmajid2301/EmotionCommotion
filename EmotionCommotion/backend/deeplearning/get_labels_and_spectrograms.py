import sys
import os
import pandas as pd
sys.path.insert(0, '../')
sys.path.insert(0, '../../app/')

from predictors import index_to_label,label_to_barray,index_to_barray
from datagrabber import get_frames, get_audiofile
from glob import glob
import numpy as np
from sklearn import preprocessing
from scipy import signal
from sklearn.decomposition import PCA
import pickle

IEMOCAP_LOCATION = '../../../../local/'
SCALER_LOCATION  = "scaler.sav"


labels = pd.read_csv('allLabels.csv')
labels = labels.drop('time',axis=1)
labels = labels.set_index('session')

X = np.zeros((40000,16000))
y = np.zeros((40000,4))
j = 0
for session in range(1,6):
    print('\n' + "Extracting from session: " + str(session) + '\n')
    numdir = len(os.listdir(IEMOCAP_LOCATION + 'IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/'))
    for i,directory in enumerate(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/')):
        sys.stdout.write("\r%d%%" % ((i/numdir)*100))
        sys.stdout.flush()
        for filename in glob(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/' + directory + '/*.wav'):
            audiofile = get_audiofile(filename,frame_size=16000)
            frames = get_frames(audiofile)
            name = filename.split('/')[-1][:-4]
            label = labels.loc[name].label
            for frame in frames:
                X[j] = frame
                # Get binary array from label
                y[j] = label_to_barray(label)
                j+=1
        print(j)

X = X[0:j]
y = y[0:j]

X_train =X[0:30242]
X_test =X[30242:]
y_train =y[0:30242]
y_test =y[30242:]

scaler = preprocessing.StandardScaler().fit(X_train)

print("Scaling train...")
X_train_preprocessed = scaler.transform(X_train)
print("Scaling test...")
X_test_preprocessed = scaler.transform(X_test)

scalerfile = SCALER_LOCATION
pickle.dump(scaler, open(scalerfile, 'wb'))

print("Getting spectograms...")

del X
del y
del X_train
del X_test

'''
X_train_spectos = np.array(list(map(lambda a: signal.spectrogram(a,nperseg=128)[2],X_train_preprocessed)))
X_test_spectos = np.array(list(map(lambda a: signal.spectrogram(a,nperseg=128)[2],X_test_preprocessed)))

print("Saving...")

np.save('../../../../local/whitened_data/X_train_no_pca.npy',X_train_spectos)
np.save('../../../../local/whitened_data/X_test_no_pca.npy',X_test_spectos)
np.save('../../../../local/whitened_data/y_train.npy',y_train)
np.save('../../../../local/whitened_data/y_test.npy',y_test)

'''

pca = PCA(n_components=60,whiten=True)

X_train_whitened = np.array(list(map(lambda a: pca.fit_transform(signal.spectrogram(a,nperseg=128)[2]),X_train_preprocessed)))
X_test_whitened = np.array(list(map(lambda a: pca.fit_transform(signal.spectrogram(a,nperseg=128)[2]),X_test_preprocessed)))

print("Sa...")

np.save('../../../../local/whitened_data/X_train_whitened_60.npy',X_train_whitened)
np.save('../../../../local/whitened_data/X_test_whitened_60.npy',X_test_whitened)
np.save('../../../../local/whitened_data/y_train.npy',y_train)
np.save('../../../../local/whitened_data/y_test.npy',y_test)
