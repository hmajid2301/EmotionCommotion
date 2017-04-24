# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:18:26 2016

@author: Olly Styles
"""
import sys
sys.path.append('../')
from datagrabber import extractAndSave
from datagrabber import extractAndSave,extractAndSaveYoutubeData

IEMOCAP_LOCATION = "../../../../local"
YOUTUBE_LOCATION = "../../../../local/wild_dataset/10_to_20_seconds"

def zerocrossing(frame, audiofile):
    '''
    Returns the number of times the signal crosses the y-axis for a given
    frame.
    '''
    n = 0
    for i in range(0, len(frame)-1):
        if (frame[i]> 0 != frame[i+1] > 0): # != is xor operator in python
            n += 1
    return [n]

# Extract energy from IEMOCAP and YouTube datasets
extractAndSave(zerocrossing, ['zerocrossing'],IEMOCAP_LOCATION,2,True,True)
extractAndSaveYoutubeData(zerocrossing,["zerocrossing"],YOUTUBE_LOCATION,2,True,True)
