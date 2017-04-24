#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:04:22 2016

@author: Tom
"""
#from importlib.machinery import SourceFileLoader

import sys
sys.path.append('../sourceFiles/')
sys.path.append('../')
import thinkplot as tp
import thinkdsp as td
from datagrabber import extractAndSave,extractAndSaveYoutubeData

IEMOCAP_LOCATION = "../../../../local"
YOUTUBE_LOCATION = "../../../../local/wild_dataset/10_to_20_seconds"


threshold=0.5
#assuming for now that thinkdsp and clip is in the same directory
def f0(frame, audiofile):
    threshold=0.5


    clip = td.Wave(frame)
    spectrum = clip.make_spectrum()

    # to find F0 we can sort by Hz, unfortunately we have small readings at 0.0Hz
    # and other small frequencies which make finding the true F0 hard

    # remove the small amplitudes which don't well describe the energy in the sound.
    # What is small is likely to be relative to the largest amplitude so use some threshold
    greatest_Hz = (spectrum.peaks()[0])[0]
    selected_pairs = []
    list_pos = 0
    while True:
        pair = (spectrum.peaks()[list_pos])
        within_threshold = (pair[0] > (greatest_Hz * threshold))
        if within_threshold == False:
            break
        selected_pairs.append(pair)
        list_pos = list_pos + 1

    # selected_pairs is now a list of the pairs which have a prominent amplitude
    sorted_by_Hz = sorted(selected_pairs, key=lambda tup: tup[1])
    # the lowest Hz should now be F0, will need to tweek 0.5
    if len(sorted_by_Hz) == 0:
        return [0]
    else:
        return [sorted_by_Hz[0][1]]

# Extract f0 from IEMOCAP and YouTube datasets
extractAndSave(f0,['f0'],IEMOCAP_LOCATION,2,True,True)
extractAndSaveYoutubeData(f0,["f0"],YOUTUBE_LOCATION,2,True,True)
