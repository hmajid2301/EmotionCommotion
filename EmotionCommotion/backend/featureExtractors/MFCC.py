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
    '''
    Computes the MEL FREQUENCY CEPSTRAL COEFFICIENTS for the frame,
    the frame is zero padded to achieve a frame lenght which is a power
    of two if this is not already the case. The power spectrum is then computed
    and this is placed into filterbanks on a mel-scale. The coefficents of
    12 of the banks is then returned.
    '''
    coefficientsCount = 12

    sampleRate = audiofile['sample_rate']
    frame_size = audiofile['frame_size']

    fftsize = pow(2, int(math.log(frame_size, 2) + 0.5)) # Round to nearest power of 2 to facilitate FFT

    
    m = aub.mfcc(fftsize, 40, coefficientsCount, sampleRate)
    
    #first we need to convert this frame to the power spectrum using a DFT
    p = aub.pvoc(fftsize, int(frame_size))
    #in order to compute DFT the frame must be of a length which is a power of 2, so expand to fftsize using zero padding
    if len(frame) != 16000:
        frame = np.pad(frame,(0,frame_size-len(frame)),'constant',constant_values=0)
    #compute the power spectrum
    spec = p(frame.astype(np.float32))
    
    #compute the MFCC, which returns the coefficents of each of the 12 coefficents 
    mfcc_out = m(spec)
    return mfcc_out



# 1. Frame the signal into short frames.
# 2. For each frame calculate the periodogram estimate of the power spectrum.
# 3. Apply the mel filterbank to the power spectra, sum the energy in each filter.
# 4. Take the logarithm of all filterbank energies.
# 5. Take the DCT of the log filterbank energies.
# 6. Keep DCT coefficients 2-13, discard the rest

#http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#computing-the-mel-filterbank

# Extract MFCC from IEMOCAP and YouTube datasets
extractAndSave(mfcc,labels,IEMOCAP_LOCATION,2,True,True)
extractAndSaveYoutubeData(mfcc,labels,YOUTUBE_LOCATION,2,True,True)
