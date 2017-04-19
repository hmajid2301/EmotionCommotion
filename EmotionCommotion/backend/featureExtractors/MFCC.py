#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 12:04:22 2016

@author: Tom
"""
import numpy as np
import aubio as  aub
import sys
sys.path.append('../')
import math

from datagrabber import extractAndSave,extractAndSaveYoutubeData

IEMOCAP_LOCATION = "../../../../local"
YOUTUBE_LOCATION = "../../../../local/wild_dataset/10_to_20_seconds"


coefficientsCount = 12
labels = ['mfcccoeff%s' % str(i) for i in range(coefficientsCount)]


def mfcc(frame, audiofile):
    coefficientsCount = 12

    sampleRate = audiofile['sample_rate']
    frame_size = audiofile['frame_size']

    fftsize = pow(2, int(math.log(frame_size, 2) + 0.5)) # Round to nearest power of 2


    m = aub.mfcc(fftsize, 40, coefficientsCount, sampleRate)
    p = aub.pvoc(fftsize, int(frame_size))
    if len(frame) != 16000:
        frame = np.pad(frame,(0,frame_size-len(frame)),'constant',constant_values=0)
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

#extractAndSave(mfcc,labels,IEMOCAP_LOCATION,2,False)
extractAndSaveYoutubeData(mfcc,labels,YOUTUBE_LOCATION,2)
