#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 12:04:22 2016

@author: Tom
"""
import scipy.io.wavfile as wav   # Reads wav file
import pandas as pd
import numpy as np
from aubio import source, pvoc, mfcc
from datagrabber import extractAndSave

IEMOCAP_LOCATION = "../../../../local"

coefficientsCount = 12
labels = ['mfcccoeff%s' % str(i) for i in range(coefficientsCount)]

sampleRate = 16000

m = mfcc(512, 40, coefficientsCount, sampleRate)
p = pvoc(512, 512/4)

def mfcc(frame, filename):
    spec = p(frame.astype(np.float32))
    
    mfcc_out = m(spec)
    return mfcc_out
    

# 1. Frame the signal into short frames.
# 2. For each frame calculate the periodogram estimate of the power spectrum.
# 3. Apply the mel filterbank to the power spectra, sum the energy in each filter.
# 4. Take the logarithm of all filterbank energies.
# 5. Take the DCT of the log filterbank energies.
# 6. Keep DCT coefficients 2-13, discard the rest

#http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#computing-the-mel-filterbank

extractAndSave(mfcc,labels,IEMOCAP_LOCATION,1)

