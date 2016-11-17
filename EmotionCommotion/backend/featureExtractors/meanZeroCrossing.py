# -*- coding: utf-8 -*-

from datagrabber import extractAndSave

IEMOCAP_LOCATION = "../../../../local"

def zerocrossing(frame, audiofile):
    i = 0
    pos = frame[0]
    for x in range(0, len(frame)-1):
        if (frame[i]*frame[i+1] > 0):
            i += 1
    return [i]

extractAndSave(zerocrossing, ['zerocrossing'],IEMOCAP_LOCATION,2)
