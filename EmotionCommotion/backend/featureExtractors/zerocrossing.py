# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from datagrabber import extractAndSave
from datagrabber import extractAndSave,extractAndSaveYoutubeData

IEMOCAP_LOCATION = "../../../../local"
YOUTUBE_LOCATION = "../../../../local/wild_dataset/10_to_20_seconds"

def zerocrossing(frame, audiofile):
    n = 0
    for i in range(0, len(frame)-1):
        if (frame[i]> 0 != frame[i+1] > 0): # != is xor operator in python
            n += 1                          #it's so ambiguous i feel the need to write a comment
    return [n]

#extractAndSave(zerocrossing, ['zerocrossing'],IEMOCAP_LOCATION,2,False)
extractAndSaveYoutubeData(zerocrossing,["zerocrossing"],YOUTUBE_LOCATION,2)
