import scipy.io.wavfile as wav   # Reads wav file
import pandas as pd
import numpy as np
import sys
sys.path.append('../')

from datagrabber import extractAndSave

IEMOCAP_LOCATION = "../../../../local"

def cepstrum(frame, filename):
    audio = np.fft.fft(frame)
    audio = abs(audio)
    audio = audio ** 2
    audio = np.log2(audio)
    audio = np.fft.ifft(audio)
    audio = audio.real
    return [np.amax(audio), np.average(audio), np.var(audio)]

extractAndSave(cepstrum,["max", "mean", "var"],IEMOCAP_LOCATION,2,False)
