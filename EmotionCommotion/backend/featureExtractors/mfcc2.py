import scipy.io.wavfile as wav   # Reads wav file
import pandas as pd
import numpy as np

from datagrabber import extractAndSave

IEMOCAP_LOCATION = "../../../../local"

def mfcc(frame, filename):
    coefficientsCount = 12
    
    minHz = 0
    maxHz = 22 #should be sample rate / 2
    
    #first compute the Periodogram estimate of the power spectrum (same as cepstrum.py)
    C = np.fft.fft(frame)
    P = abs(S)
    P = S**2
    
    #now we compute the filter banks
    print frame
    
    return None

extractAndSave(mfcc,["max", "mean", "var"],IEMOCAP_LOCATION,2)
 
