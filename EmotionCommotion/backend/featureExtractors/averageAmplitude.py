import scipy.io.wavfile as wav   # Reads wav file

from datagrabber import extractAndSave

IEMOCAP_LOCATION = "../../../../local"

def averageAmplitude(filename):
    audio = wav.read(filename)
    return abs(audio[1]).mean()
    
extractAndSave(averageAmplitude,IEMOCAP_LOCATION,2)
