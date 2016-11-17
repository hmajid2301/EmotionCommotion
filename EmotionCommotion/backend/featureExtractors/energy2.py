#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:14:44 2016

@author: olly
"""

import librosa
from datagrabber import extractAndSave
import numpy as np

IEMOCAP_LOCATION = "../../../../local"

def energy(filename):
    audio,sr = librosa.load(filename,sr=None)
    return sum(np.apply_along_axis(lambda x: x**2,0,audio))

extractAndSave(energy,IEMOCAP_LOCATION,2,5,False)