# -*- coding: utf-8 -*-

import numpy as np
from datagrabber import extractAndSave


IEMOCAP_LOCATION = "../../../../local"

def amplitude(frame):
    return [np.amax(frame), np.average(frame),np.var(frame)]
    
extractAndSave(amplitude,["max", "mean","var"],IEMOCAP_LOCATION,2)