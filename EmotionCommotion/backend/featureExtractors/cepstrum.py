import scipy.io.wavfile as wav   # Reads wav file
import pandas as pd
import numpy as np
import sys
sys.path.append('../')

from datagrabber import extractAndSave,extractAndSaveYoutubeData

IEMOCAP_LOCATION = "../../../../local"
YOUTUBE_LOCATION = "../../../../local/wild_dataset/10_to_20_seconds"

def cepstrum(frame, filename):
    audio = np.fft.fft(frame)
    audio = abs(audio)
    audio = audio ** 2
    audio = np.log2(audio)
    audio = np.fft.ifft(audio)
    audio = audio.real
    return [np.amax(audio), np.average(audio), np.var(audio)]

#extractAndSave(cepstrum,["max", "mean", "var"],IEMOCAP_LOCATION,2,False)
extractAndSaveYoutubeData(cepstrum,["max", "mean","var"],YOUTUBE_LOCATION,2)
