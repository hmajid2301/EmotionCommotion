#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:18:26 2016

@author: Olly Styles
"""
# In[1]
import scipy.io.wavfile as wav   # Reads wav file
import pandas as pd
import numpy as np
import os
import glob
import sys
sys.path.append('../')

from datagrabber import extractAndSave,extractAndSaveYoutubeData

# In[2]
IEMOCAP_LOCATION = "../../../../local"
YOUTUBE_LOCATION = "../../../../local/wild_dataset/10_to_20_seconds"


last_filename = "filename"

def silence_ratio(frame, audiofile):
    '''
    Returns the ratio of time signal was above a threshold to the time below.
    The threshold value is defined in the audiofile object
    '''
    threshold = audiofile['threshold']
    thresholded_frame = frame[np.where(abs(frame) > threshold)]
    ratio = 1 - (len(thresholded_frame) / len(frame))
    return [ratio]

# Extract silence ratio from IEMOCAP and YouTube datasets
extractAndSave(silence_ratio,["silence_ratio"],IEMOCAP_LOCATION,2,True,True)
extractAndSaveYoutubeData(silence_ratio,["silence_ratio"],YOUTUBE_LOCATION,2,True,True)
