from __future__ import division
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav   # Reads wav file
import os
import sys
from glob import glob
from datagrabber import *
from sklearn import preprocessing
import pickle

IEMOCAP_LOCATION = "../../../local"
SCALER_LOCATION  = "frame_scaler.sav"
verbose = 2

# Initialise matrix
allframes = np.zeros((100000,16000))

j = 0
# Get all frames from all sessions in IEMOCAP dataset
for session in range(1,6):
    if verbose > 0:
        print('\n' + "Extracting from session: " + str(session) + '\n')
        numdir = len(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/'))
    for i,directory in enumerate(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/')):
        if verbose > 1:
            sys.stdout.write("\r%d%%" % ((i/numdir)*100))
            sys.stdout.flush()
        for filename in glob(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/' + directory + '/*.wav'):
            audio = get_audiofile(filename,frame_size=16000)
            frames = get_frames(audio)
            for f in frames:
                allframes[j] = f
                j += 1

# Resize matrix
allframes = allframes[0:j]

# Fit scalar to frames
scaler = preprocessing.StandardScaler().fit(allframes)

# Save scalar
pickle.dump(scaler, open(SCALER_LOCATION, 'wb'))
