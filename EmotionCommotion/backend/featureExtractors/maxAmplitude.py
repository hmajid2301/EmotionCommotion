import scipy.io.wavfile as wav   # Reads wav file

from datagrabber import extractAndSave

IEMOCAP_LOCATION = "../../../../local"

def maxAmplitude(filename):
    audio = wav.read(filename)
    return abs(audio[1]).max()
    
extractAndSave(maxAmplitude,IEMOCAP_LOCATION,2)