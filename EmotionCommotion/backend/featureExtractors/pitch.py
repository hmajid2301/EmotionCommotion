#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:04:22 2016

@author: Tom
"""
import sys
sys.path.append('../sourceFiles/')
import thinkplot as tp
import thinkdsp as td
sys.path.append('../')

from datagrabber import extractAndSave,extractAndSaveYoutubeData

IEMOCAP_LOCATION = "../../../../local"
YOUTUBE_LOCATION = "../../../../local/wild_dataset/10_to_20_seconds"

#make_spectrum is sorted in order of decreasing amplitude, so the first pair is going to be the amplitude of the pitch and the pitch frequency.
def pitch(frame, audiofile):
    clip = td.Wave(frame)
    spectrum = clip.make_spectrum()
    return spectrum.peaks()[0][1]

#extractAndSave(pitch,['pitch'],IEMOCAP_LOCATION,2,False)
extractAndSaveYoutubeData(pitch,["pitch"],YOUTUBE_LOCATION,2)
