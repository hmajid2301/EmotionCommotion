from __future__ import division
import scipy.io.wavfile as wav   # Reads wav file
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

IEMOCAP_LOCATION = "../../../../local"
SCALER_LOCATION  = "../../frame_scaler.sav"

# Load pre-trained scaler
scaler = pickle.load(open(SCALER_LOCATION, 'rb'))

# Aggregation functions
agg_funcs = [np.amax,np.average,np.var]
agg_func_names = ["max", "mean", "var"]

def get_frames(audiofile,standardize_frame=False):
    '''
    Splits an audiofile into frames. The audiofile object specifies the
    frame size. If standardize_frame is set to true, frames will be standardized
    using the scaler file. Returns a list of frames.
    '''
    # Get data from audiofile object
    frame_size = audiofile['frame_size']
    frame_overlap = audiofile['frame_overlap']
    specto_tresh = audiofile['specto_thres']
    audio = audiofile['audio']
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = audio[:,0]
    frames = []
    i = 0
    while ((i+1)*(frame_size - frame_overlap) < len(audio)):
        # Get frame
        frame = np.array(audio[i*(frame_size - frame_overlap):(i+2)*(frame_size - frame_overlap)])
        if len(frame) != frame_size:
            # Pad frame if necessary
            frame = np.pad(frame, (0, frame_size - len(frame)%frame_size), 'constant')
        if standardize_frame:
            # Use scaler to standardize frame
            frame = scaler.transform(frame.reshape(1,-1))
        frames.append(frame[0])
        i += 1
    return frames

def aggregate(vals):
    '''
    Apply each aggregation function to a list of values. Returns a list of
    aggregated values
    '''
    agg_vals = []
    for i in range(0, len(agg_funcs)):
        agg_vals = np.concatenate((agg_vals,agg_funcs[i](vals, axis=0)), axis=0)
    return agg_vals

def get_audiofile(filename, data=None,read_file=True,frame_size=16000):
    '''
    Given the filename of a wav file, reads the file and returns an audiofile
    dictionary containing information about the file. If read_file is set to
    false, the audiofile dictionary will be created using the data argument
    '''
    audiofile = {}
    sample_rate = 32000
    if (read_file):
        [sample_rate, audio] = wav.read(filename)
    else:
        audio = data

    audiofile['sample_rate'] = sample_rate
    audiofile['frame_size'] = frame_size
    audiofile['filename'] = filename
    audiofile['overlap_ratio'] = 0.5
    audiofile['frame_overlap'] = int(audiofile['frame_size'] * audiofile['overlap_ratio'])
    audiofile['audio'] = audio
    audiofile['threshold'] = np.max(audio) * 0.03
    audiofile['specto_thres'] = np.max(audio) * 0.1
    return audiofile

def extractAndSave(funct,labels,IEMOCAP_LOCATION,verbose=1,aggregate_vals=True,standardize_frame=False):
    '''
    Expects a function of the form func(filename)
    Applies a feature extraction function to all wav files
    in the IMEOCAP database, and saves the results
    in the feaures directory.
    '''
    dic = {}
    vals = []
    # Loop through all files in IMEOCAP dataset
    for session in range(1,6):
        if verbose > 0:
            print('\n' + "Extracting from session: " + str(session) + '\n')
            numdir = len(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/'))
        for i,directory in enumerate(os.listdir(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/')):
            if verbose > 1:
                sys.stdout.write("\r%d%%" % ((i/numdir)*100))
                sys.stdout.flush()
            for filename in glob(IEMOCAP_LOCATION + '/IEMOCAP_full_release/Session' + str(session) + '/sentences/wav/' + directory + '/*.wav'):
                name = filename.split('/')[-1][:-4] # Extract filename
                audiofile = get_audiofile(filename)
                frames = get_frames(audiofile,standardize_frame) # Split into frames
                if aggregate_vals:
                    vals = []
                    for frame in frames:
                        vals.append(funct(frame, audiofile)) # Apply function to each frame
                    agg_vals = aggregate(vals) # Aggregate values
                    dic[name] = agg_vals
                else:
                    for frame in frames:
                        vals.append(funct(frame, audiofile))


    # Save results
    if aggregate_vals:
        df = pd.DataFrame.from_dict(dic,orient='index').reset_index()
        columns = ['session']
        for i in range(0, len(agg_func_names)):
            for j in range(0, len(labels)):
                columns.append(agg_func_names[i]+'('+labels[j]+'('+funct.__name__+'))')
        df.columns = columns
        df = df.sort_values(by='session')
        df.to_csv('../features/' + funct.__name__ + '_standardized.csv',index=False)
    else:
        vals = np.array(vals)
        np.save('../features/' + funct.__name__ + '_framewise.npy',vals)

def preprocess_frame(frame):
    '''
    Scales, computes spectogram and applys PCA on a frame
    '''
    scaled_frame = scaler.transform(frame)
    spectogram = signal.spectrogram(scaled_frame,nperseg=128)[2]
    pca_spectogram = np.array(pca.fit_transform(spectogram))
    return pca_spectogram

def extractAndSaveYoutubeData(funct,labels,data_location,verbose=1,aggregate_vals=True,standardize_frame=False):
    '''
    Expects a function of the form func(filename)
    Applies a feature extraction function to all wav files
    in the youtube database, and saves the results
    in the feaures directory.
    '''
    dic = {}
    vals = []
    num_files = len(glob(data_location + '/*.wav'))
    i = 0
    # Loop through all files in IMEOCAP dataset
    for filename in glob(data_location + '/*.wav'):
        if verbose > 1:
            sys.stdout.write("\r%d%%" % ((i/num_files)*100))
            sys.stdout.flush()
        name = filename.split('/')[-1][:-4]             # Extract filename
        audiofile = get_audiofile(filename,frame_size=16000)
        frames = get_frames(audiofile,standardize_frame)    # Split into frames
        if aggregate_vals:
            vals = []
            for frame in frames:
                vals.append(funct(frame, audiofile))   # Apply function to each frame
            agg_vals = aggregate(vals)      # Aggregate values
            dic[name] = agg_vals
        else:
            for frame in frames:
                vals.append(funct(frame, audiofile))
        i = i+1
    # Save results
    print("")
    if aggregate_vals:
        df = pd.DataFrame.from_dict(dic,orient='index').reset_index()
        columns = ['session']
        for i in range(0, len(agg_func_names)):
            for j in range(0, len(labels)):
                columns.append(agg_func_names[i]+'('+labels[j]+'('+funct.__name__+'))')

        df.columns = columns
        df = df.sort_values(by='session')
        df.to_csv('../features/youtube_' + funct.__name__ + '_standardized.csv',index=False)
    else:
        vals = np.array(vals)
        vals_train = vals[0:30242]
        vals_test = vals[30242:]
        np.save('../features/youtube_' + funct.__name__ + '_framewise_train.npy',vals_train)
        np.save('../features/youtube_' + funct.__name__ + '_framewise_test.npy',vals_test)
