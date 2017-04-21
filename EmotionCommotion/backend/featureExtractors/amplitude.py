# -*- coding: utf-8 -*-

import numpy as np
import sys
sys.path.append('../')
from datagrabber import extractAndSave,extractAndSaveYoutubeData



IEMOCAP_LOCATION = "../../../../local"
YOUTUBE_LOCATION = "../../../../local/wild_dataset/10_to_20_seconds"

def amplitude(frame,filename):
    return [np.amax(frame), np.average(frame),np.var(frame)]

extractAndSave(amplitude,["max", "mean","var"],IEMOCAP_LOCATION,2,True,True)
extractAndSaveYoutubeData(amplitude,["max", "mean","var"],YOUTUBE_LOCATION,2,True,True)
