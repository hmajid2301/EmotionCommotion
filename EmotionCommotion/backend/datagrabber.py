import scipy.io.wavfile as wav   # Reads wav file


IEMOCAP_LOCATION = "../../../../local"
SCALER_LOCATION  = "deeplearning/scaler.sav"

import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from glob import glob
import sys
from types import *
from sklearn.decomposition import PCA
from scipy import signal
import pickle

#scalerfile = SCALER_LOCATION
#scaler = pickle.load(open(scalerfile, 'rb'))

def get_frames(audiofile):
    frame_size = audiofile['frame_size']
    frame_overlap = audiofile['frame_overlap']
    specto_tresh = audiofile['specto_thres']
    audio = audiofile['audio']
    frames = []
    i = 0
    while ((i+1)*(frame_size - frame_overlap) < len(audio)):
        frame = np.array(audio[i*(frame_size - frame_overlap):(i+2)*(frame_size - frame_overlap)])
        if len(frame) != frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)%frame_size), 'constant')
        #Ensure there is a value above threshold
        #mx = np.amax(abs(frame))
        #if mx > specto_tresh:
        frames.append(frame)
        i += 1
    return frames

agg_funcs = [np.amax,np.average,np.var]
agg_func_names = ["max", "mean", "var"]

def aggregate(vals):
    agg_vals = []
    for i in range(0, len(agg_funcs)):
        agg_vals = np.concatenate((agg_vals,agg_funcs[i](vals, axis=0)), axis=0)
    return agg_vals

def get_audiofile(filename, data=None,flag=True,frame_size=32000):
    audiofile = {}
    sample_rate = 32000
    if (flag):
        [sample_rate, audio] = wav.read(filename)
    else:
        audio = data

    audiofile['sample_rate'] = sample_rate
    audiofile['frame_size'] = frame_size
    audiofile['filename'] = filename
    audiofile['overlap_ratio'] = 0.5
    audiofile['frame_overlap'] = int(audiofile['frame_size'] * audiofile['overlap_ratio'])
    audiofile['audio'] = audio
    audiofile['threshold'] = max(audio) * 0.03
    audiofile['specto_thres'] = max(audio) * 0.1
    return audiofile

def extractAndSave(funct,labels,IEMOCAP_LOCATION,verbose=1):
    '''
    Expects a function of the form func(filename)
    Applies a feature extraction function to all wav files
    in the IMEOCAP database, and saves the results
    in the feaures directory.
    '''

    # Fill a dict with values
    dic = {}
    for session in range(1,6):
        if verbose > 0:
            print('\n' + "Extracting from session: " + str(session) + '\n')
            numdir = len(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/'))
        for i,directory in enumerate(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/')):
            if verbose > 1:
                sys.stdout.write("\r%d%%" % ((i/numdir)*100))
                sys.stdout.flush()
            for filename in glob(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/' + directory + '/*.wav'):
                name = filename.split('/')[-1][:-4]
                audiofile = get_audiofile(filename)
                frames = get_frames(audiofile)
                vals = []
                for frame in frames:
                    vals.append(funct(frame, audiofile))
                agg_vals = aggregate(vals)
                dic[name] = agg_vals

    # Save results
    df = pd.DataFrame.from_dict(dic,orient='index').reset_index()
    columns = ['session']
    for i in range(0, len(agg_func_names)):
        for j in range(0, len(labels)):
            columns.append(agg_func_names[i]+'('+labels[j]+'('+funct.__name__+'))')
    df.columns = columns
    #df.columns = ['session',funct.__name__+"max-max", funct.__name__+"mean-max",funct.__name__+"max-mean", funct.__name__+"max-var", funct.__name__+"mean-mean", funct.__name__+"mean-var"]
    df = df.sort_values(by='session')
    df.to_csv('../features/' + funct.__name__ + '.csv',index=False)

def preprocess_frame(frame):
    scaled_frame = scaler.transform(frame)
    spectogram = signal.spectrogram(scaled_frame,nperseg=128)[2]
    pca_spectogram = np.array(pca.fit_transform(spectogram))
    return pca_spectogram
