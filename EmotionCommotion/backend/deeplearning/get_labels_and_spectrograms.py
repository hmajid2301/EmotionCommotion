from __future__ import division
import sys
import os
import pandas as pd
from glob import glob
import numpy as np
from sklearn import preprocessing
from scipy import signal
from sklearn.decomposition import PCA

# Add to system path so local files can be located
sys.path.insert(0, '../')
sys.path.insert(0, '../../app/')

from predictors import index_to_label,label_to_barray,index_to_barray
from datagrabber import get_frames, get_audiofile


IEMOCAP_LOCATION = '../../../../local/'
YOUTUBE_LOCATION = '../../../../local/wild_dataset/10_to_20_seconds'
SCALER_LOCATION  = "../frame_scaler.sav"

# Read labels for IEMOCAP database
iemo_labels = pd.read_csv('../data/allLabels.csv')
iemo_labels = iemo_labels.drop('time',axis=1)
iemo_labels = iemo_labels.set_index('session')

# Initialise matrices for features (X) and labels (y)
iemo_X = np.zeros((40000,16000))
iemo_y = np.zeros((40000,4))
j = 0

# Split every file in the IEMOCAP database into frames and save frame-label for
# each frame
for session in range(1,6):
    print('\n' + "Extracting from session: " + str(session) + '\n')
    numdir = len(os.listdir(IEMOCAP_LOCATION + 'IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/'))
    for i,directory in enumerate(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/')):
        sys.stdout.write("\r%d%%" % ((i/numdir)*100))
        sys.stdout.flush()
        for filename in glob(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/' + directory + '/*.wav'):
            audiofile = get_audiofile(filename,frame_size=16000)
            frames = get_frames(audiofile,True)
            name = filename.split('/')[-1][:-4]    # Get session name
            label = iemo_labels.loc[name].label    # Find label
            for frame in frames:
                iemo_X[j] = frame
                # Get binary array from label
                iemo_y[j] = label_to_barray(label)
                j+=1
    print(j)
# Resize matrices
iemo_X = iemo_X[0:j]
iemo_y = iemo_y[0:j]


print("Getting iemocap spectograms...")

pca = PCA(n_components=40,whiten=True)

# Generate spectrogram for each frame and reduce dimensionality using PCA
iemo_X_specto = np.array(list(map(lambda a: pca.fit_transform(signal.spectrogram(a,nperseg=128)[2]),iemo_X)))

print("Saving iemocap...")

# Save spectrograms and frame-labels for IEMOCAP dataset
np.save('../../../../local/whitened_data/iemo_X_whitened_40.npy',iemo_X_specto)
np.save('../../../../local/whitened_data/iemo_y.npy',iemo_y)

# Read labels for YouTube database
yt_labels = pd.read_csv('../data/emmotion_labels.csv')
yt_labels = yt_labels.set_index('session')

# Initialise matrices for features (X) and labels (y)
yt_X = np.zeros((40000,16000))
yt_y = np.zeros((40000,4))
num_files = len(glob(YOUTUBE_LOCATION + '/*.wav'))

j = 0
# Split every file in the YouTube database into frames and save frame-label for
# each frame
for filename in glob(YOUTUBE_LOCATION + '/*.wav'):
    sys.stdout.write("\r%d%%" % ((j/num_files)*100))
    sys.stdout.flush()
    audiofile = get_audiofile(filename,frame_size=16000)
    frames = get_frames(audiofile,True)
    name = filename.split('/')[-1][:-4]
    label = yt_labels.loc[name].emotion
    for frame in frames:
        yt_X[j] = frame
        # Get binary array from label
        yt_y[j] = label_to_barray(label)
    j+=1

# Resize matrices
yt_X = yt_X[0:j]
yt_y = yt_y[0:j]

print("Getting youtube spectograms...")

# Generate spectrogram for each frame and reduce dimensionality using PCA
yt_X_specto = np.array(list(map(lambda a: pca.fit_transform(signal.spectrogram(a,nperseg=128)[2]),yt_X)))

print("Saving youtube...")

# Save spectrograms and frame-labels for YouTube dataset
np.save('../../../../local/whitened_data/yt_X_whitened_40.npy',yt_X_specto)
np.save('../../../../local/whitened_data/yt_y.npy',yt_y)
