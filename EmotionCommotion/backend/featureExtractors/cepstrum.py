import scipy.io.wavfile as wav   # Reads wav file
import pandas as pd
import numpy as np

from datagrabber import extractAndSave

IEMOCAP_LOCATION = "../../../../local"

def cepstrum(filename):
    audio = pd.DataFrame(wav.read(filename)[1])
    audio = np.fft.fft(audio)
    audio = abs(audio)
    audio = audio ** 2
    audio = np.log2(audio)
    audio = np.fft.ifft(audio)
    return [np.amax(audio), np.average(audio), np.var(audio)]

extractAndSave(cepstrum,["max", "mean", "var"],IEMOCAP_LOCATION,2)
