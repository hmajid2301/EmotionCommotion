#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:25:14 2016

@author: olly
"""

# -*- coding: utf-8 -*-

import librosa
from datagrabber import extractAndSave

IEMOCAP_LOCATION = "../../../../local"

def varZeroCrossing(filename):
    audio,sr = librosa.load(filename,sr=None)
    return librosa.feature.zero_crossing_rate(audio).varr()

extractAndSave(varZeroCrossing,IEMOCAP_LOCATION,2,5,False)