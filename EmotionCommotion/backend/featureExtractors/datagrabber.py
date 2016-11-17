import scipy.io.wavfile as wav   # Reads wav file


IEMOCAP_LOCATION = "../../../../local"

import numpy as np
import pandas as pd
import os
from glob import glob
import sys


def get_frames(filename):
    [sample_rate, audio] = wav.read(filename)
    frame_size = sample_rate // 5
    overlap_ratio = 0.5
    frame_overlap = int(frame_size * overlap_ratio)
    frames = []
    i = 0
    while ((i+1)*(frame_size - frame_overlap) < len(audio)):
        frames.append(audio[i*(frame_size - frame_overlap):(i+1)*(frame_size - frame_overlap)])
        i += 1
    return frames

agg_funcs = [np.amax,np.average,np.var]
agg_func_names = ["max", "mean", "var"]

def aggregate(vals):
    agg_vals = []
    for i in range(0, len(agg_funcs)):
        agg_vals = np.concatenate((agg_vals,agg_funcs[i](vals, axis=0)), axis=0)
    return agg_vals
                     

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
                frames = get_frames(filename)
                vals = []
                for frame in frames:
                    vals.append(funct(frame))
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

