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

from datagrabber import extractAndSave

# In[2]
IEMOCAP_LOCATION = "../../../../local"

last_filename = "filename"

def silence_ratio(frame, audiofile):
    threshold = audiofile['threshold']
    thresholded_frame = frame[np.where(abs(frame) > threshold)]
    ratio = 1 - (len(thresholded_frame) / len(frame))
    return [ratio]

extractAndSave(silence_ratio,["silence_ratio"],IEMOCAP_LOCATION,2,False)
