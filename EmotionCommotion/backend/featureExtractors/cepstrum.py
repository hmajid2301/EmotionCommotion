import scipy.io.wavfile as wav   # Reads wav file
import pandas as pd
import numpy as np
import sys
sys.path.append('../')

from datagrabber import extractAndSave,extractAndSaveYoutubeData

IEMOCAP_LOCATION = "../../../../local"
YOUTUBE_LOCATION = "../../../../local/wild_dataset/10_to_20_seconds"

def cepstrum(frame, filename):
    '''
    Computes the cepstrum of the frame, which is calculated by converting 
    the power spectrum (|DCT|^2) of the audio frame to a log-scale.
    The max average and variance of the resulting audio clip are returned
    as features
    '''
    #get DCT
    audio = np.fft.fft(frame)
    #get |DCT|
    audio = abs(audio)
    #get |DCT|^2, i.e. the power spectrum
    audio = audio ** 2
    #take a log, which simulates the non-linearality of human hearing
    audio = np.log2(audio)
    #take the real of the IDFT as the feature
    audio = np.fft.ifft(audio)
    audio = audio.real
    return [np.amax(audio), np.average(audio), np.var(audio)]

# Extract cepstrum from IEMOCAP and YouTube datasets
extractAndSave(cepstrum,["max", "mean", "var"],IEMOCAP_LOCATION,2,True,True)
extractAndSaveYoutubeData(cepstrum,["max", "mean","var"],YOUTUBE_LOCATION,2,True,True)
