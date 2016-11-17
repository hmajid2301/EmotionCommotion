# -*- coding: utf-8 -*-

import librosa
from datagrabber import extractAndSave

IEMOCAP_LOCATION = "../../../../local"

def meanZeroCrossing(filename):
    audio,sr = librosa.load(filename,sr=None)
    return librosa.feature.zero_crossing_rate(audio).mean()

extractAndSave(meanZeroCrossing,IEMOCAP_LOCATION,2,5,False)