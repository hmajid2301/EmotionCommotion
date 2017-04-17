#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:04:22 2016

@author: Tom
"""
import sys
sys.path.append('../')

from datagrabber import extractAndSave,extractAndSaveYoutubeData
import numpy as np

IEMOCAP_LOCATION = "../../../../local"
YOUTUBE_LOCATION = "../../../../local/wild_dataset/10_to_20_seconds"


def energy(frame, audiofile):
    return [sum(np.apply_along_axis(lambda x: x**2,0,frame))]

#extractAndSave(energy,['energy'],IEMOCAP_LOCATION,2,False)
extractAndSaveYoutubeData(energy,["energy"],YOUTUBE_LOCATION,2)
